# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.spec_decode.ngram_proposer import (
    NgramProposer,
    _find_longest_matched_ngram_and_propose_tokens,
)
from vllm.v1.spec_decode.ngram_proposer_gpu import (
    NgramPredictionGPUKernel,
    NgramPredictionState,
    NgramProposerGPU,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState


@pytest.fixture
def prediction_kernel():
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", max_model_len=32),
        speculative_config=SpeculativeConfig(
            method="ngram_gpu",
            num_speculative_tokens=4,
            prompt_lookup_min=2,
            prompt_lookup_max=4,
        ),
    )
    with set_current_vllm_config(config):
        return NgramPredictionGPUKernel(vllm_config=config)


@pytest.mark.parametrize(
    "output,corpus,expected",
    [
        ([], [1, 2, 3, 4], []),
        ([1], [1, 2, 3, 4], []),
        ([1, 2], [1, 2], []),
        ([1, 2], [8, 9, 10], []),
        ([1, 2], [1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6]),
        ([1, 2, 3, 4], [1, 2, 3, 4, 5], [5]),
        ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], []),
        # Recompute after an insertion, deletion, or replacement.
        ([1, 2, 99, 3, 4], [1, 2, 3, 4, 5, 6], [5, 6]),
        ([1, 4, 5], [1, 2, 3, 4, 5, 6], [6]),
        ([1, 99, 3, 4], [1, 2, 3, 4, 5, 6], [5, 6]),
        # Prefer the longest match, then its earliest occurrence.
        ([1, 2, 3], [2, 3, 8, 1, 2, 3, 9], [9]),
        ([1, 2], [1, 2, 3, 1, 2, 4], [3, 1, 2, 4]),
    ],
)
def test_prediction_matches_output_only(prediction_kernel, output, corpus, expected):
    """Prompt tokens and corpus padding must not create prediction matches."""
    prompt = [1, 2, 3, 4, 1]
    tokens = torch.zeros((1, 32), dtype=torch.int32)
    tokens[0, : len(prompt) + len(output)] = torch.tensor(prompt + output)
    prediction = torch.zeros_like(tokens)
    prediction[0, : len(corpus)] = torch.tensor(corpus)
    drafts, counts = prediction_kernel.forward(
        torch.tensor([len(prompt) + len(output)]),
        tokens,
        torch.tensor([True]),
        prediction,
        torch.tensor([len(corpus)]),
        torch.tensor([len(prompt)]),
        torch.tensor([32]),
    )
    assert drafts.tolist() == [expected + [-1] * (4 - len(expected))]
    assert counts.tolist() == [len(expected)]


def test_prediction_mixed_batch_masks_and_limits(prediction_kernel):
    """Keep ordinary drafts and mask discarded or finished prediction requests."""
    tokens = torch.zeros((5, 32), dtype=torch.int32)
    tokens[:, :5] = torch.tensor([1, 2, 3, 1, 2])
    prediction = torch.zeros_like(tokens)
    prediction[:, :6] = torch.tensor([1, 2, 8, 9, 10, 11])
    drafts, counts = prediction_kernel.forward(
        torch.tensor([5] * 5),
        tokens,
        torch.tensor([True, True, False, True, True]),
        prediction,
        torch.tensor([6, 0, 6, 6, 6]),
        torch.tensor([3] * 5),
        torch.tensor([6, 32, 32, 5, 32]),
    )
    assert drafts.tolist() == [
        [8, -1, -1, -1],
        [3, 1, 2, -1],
        [-1] * 4,
        [-1] * 4,
        [8, 9, 10, 11],
    ]
    assert counts.tolist() == [1, 3, 0, 0, 4]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_prediction_state_batch_lifecycle(device):
    """Swaps, preemption, cancellation, and reused IDs cannot leak another corpus."""

    def request(req_id, tokens):
        return CachedRequestState(
            req_id=req_id,
            prompt_token_ids=[50, 51, 52],
            mm_features=[],
            sampling_params=SamplingParams(prediction_token_ids=tokens, max_tokens=7),
            generator=None,
            block_ids=([],),
            num_computed_tokens=0,
            output_token_ids=[],
        )

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    state = NgramPredictionState(3, 32, torch.device(device))
    requests = {"a": request("a", [1, 2, 3]), "b": request("b", [8, 9])}
    state.update({"a": 0, "b": 1}, requests)
    state.update({"a": 1, "b": 0}, requests)
    assert state.token_ids[0, :2].tolist() == [8, 9]
    assert state.token_ids[1, :3].tolist() == [1, 2, 3]
    assert state.lengths.tolist() == [2, 3, 0]
    assert state.prompt_lengths.tolist() == [3, 3, 0]
    assert state.token_limits.tolist() == [10, 10, 0]
    state.update({"a": 0}, requests)
    assert state.lengths.tolist() == [3, 0, 0]
    state.update({"a": 0, "b": 1}, requests)
    assert state.token_ids[1, :2].tolist() == [8, 9]
    requests["a"] = request("a", [6])
    state.update({"a": 0, "b": 1}, requests)
    assert state.token_ids[0, 0].item() == 6
    assert state.lengths.tolist() == [1, 2, 0]
    requests["a"] = request("a", None)
    state.update({"a": 0}, requests)
    assert state.lengths.tolist() == [0, 0, 0]
    assert not state.req_indices


def test_prediction_default_match_requires_five_output_tokens(prediction_kernel):
    prediction_kernel.min_n = prediction_kernel.max_n = 5
    tokens = torch.tensor([[90, 1, 2, 3, 4, 5, 0, 0]], dtype=torch.int32)
    corpus = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0]], dtype=torch.int32)
    for output_len, corpus_len, expected in (
        (4, 6, [-1] * 4),
        (5, 5, [-1] * 4),
        (5, 6, [6, -1, -1, -1]),
    ):
        drafts, _ = prediction_kernel.forward(
            torch.tensor([1 + output_len]),
            tokens,
            torch.tensor([True]),
            corpus,
            torch.tensor([corpus_len]),
            torch.tensor([1]),
            torch.tensor([8]),
        )
        assert drafts.tolist() == [expected]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_prediction_compiled_proposer_mixed_batches(monkeypatch):
    """Exercise warmup, dynamic batches, and accepted-token writes on the GPU."""
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    config = VllmConfig(
        model_config=ModelConfig(model="facebook/opt-125m", max_model_len=32),
        scheduler_config=SchedulerConfig(max_num_seqs=8, max_model_len=32),
        speculative_config=SpeculativeConfig(
            method="ngram_gpu",
            num_speculative_tokens=4,
            prompt_lookup_min=2,
            prompt_lookup_max=4,
        ),
    )
    device = torch.device("cuda:0")
    with set_current_vllm_config(config):
        proposer = NgramProposerGPU(config, device)
    request = CachedRequestState(
        req_id="prediction",
        prompt_token_ids=[50, 51, 52],
        mm_features=[],
        sampling_params=SamplingParams(
            prediction_token_ids=[1, 2, 8, 9], max_tokens=20
        ),
        generator=None,
        block_ids=([],),
        num_computed_tokens=3,
        output_token_ids=[1],
    )
    for batch_size in (1, 8, 2, 1):
        tokens = torch.zeros((batch_size, 32), dtype=torch.int32, device=device)
        tokens[:, :4] = torch.tensor([1, 2, 3, 1], device=device)
        tokens[0, :4] = torch.tensor([50, 51, 52, 1], device=device)
        proposer.prediction_state.update({"prediction": 0}, {"prediction": request})
        lengths = torch.full((batch_size,), 4, dtype=torch.int32, device=device)
        sampled = torch.full((batch_size, 5), -1, dtype=torch.int32, device=device)
        sampled[:, 0] = 2
        drafts, counts = proposer.propose(
            4,
            lengths,
            tokens,
            sampled,
            torch.ones_like(lengths),
        )
        assert drafts[0].tolist() == [8, 9, -1, -1]
        assert counts[0].item() == 2
        if batch_size > 1:
            assert drafts[1:].tolist() == [[3, 1, 2, -1]] * (batch_size - 1)
        assert lengths.tolist() == [4] * batch_size
        assert tokens[:, 4].tolist() == [2] * batch_size
    proposer.prediction_state.update({}, {})
    drafts, _ = proposer.propose(4, lengths, tokens, sampled, torch.ones_like(lengths))
    assert drafts.tolist() == [[-1, -1, -1, -1]]


def test_find_longest_matched_ngram_and_propose_tokens():
    tokens = np.array([1, 2, 3, 4, 1, 2, 3, 5, 6])
    result = _find_longest_matched_ngram_and_propose_tokens(
        origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
    )
    assert len(result) == 0

    tokens = np.array([1, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([4, 1]),
    )

    tokens = np.array([1, 3, 6, 2, 3, 4, 1, 2, 3])
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=2, max_ngram=2, max_model_len=1024, k=3
        ),
        np.array([4, 1, 2]),
    )
    # Return on the first match
    np.testing.assert_array_equal(
        _find_longest_matched_ngram_and_propose_tokens(
            origin_tokens=tokens, min_ngram=1, max_ngram=1, max_model_len=1024, k=2
        ),
        np.array([6, 2]),
    )


def test_ngram_proposer():
    def get_ngram_proposer(min_n: int, max_n: int, k: int) -> NgramProposer:
        # Dummy model config. Just to set max_model_len.
        model_config = ModelConfig(model="facebook/opt-125m")
        return NgramProposer(
            vllm_config=VllmConfig(
                model_config=model_config,
                speculative_config=SpeculativeConfig(
                    prompt_lookup_min=min_n,
                    prompt_lookup_max=max_n,
                    num_speculative_tokens=k,
                    method="ngram",
                ),
            )
        )

    # No match.
    token_ids_cpu = np.array([[1, 2, 3, 4, 5]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # No match for 4-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=4, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # No match for 4-gram but match for 3-gram.
    token_ids_cpu = np.array([[1, 2, 3, 4, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[4, 1]]))

    # Match for both 4-gram and 3-gram.
    # In this case, the proposer should return the 4-gram match.
    token_ids_cpu = np.array([[2, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=3, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 1]]

    # Match for 2-gram and 3-gram, but not 4-gram.
    token_ids_cpu = np.array([[3, 4, 5, 2, 3, 4, 1, 2, 3, 4]])
    result = get_ngram_proposer(min_n=2, max_n=4, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[1, 2]]))  # Not [5, 2]]

    # Multiple 3-gram matched, but always pick the first one.
    token_ids_cpu = np.array([[1, 2, 3, 100, 1, 2, 3, 200, 1, 2, 3, 300, 1, 2, 3]])
    result = get_ngram_proposer(min_n=3, max_n=3, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert np.array_equal(result, np.array([[100, 1]]))

    # check empty input
    token_ids_cpu = np.array([[]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0]],
        num_tokens_no_spec=np.array([len(c) for c in token_ids_cpu]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 0

    # check multibatch input
    # first request has 5 tokens and a match
    # second request has 3 tokens and no match. Padded with -1 for max len 5
    token_ids_cpu = np.array([[1, 2, 3, 1, 2], [4, 5, 6, -1, -1]])
    result = get_ngram_proposer(min_n=2, max_n=2, k=2).propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([5, 3]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([3, 1]))
    assert np.array_equal(result[1], np.array([]))

    # Test non-contiguous indices: requests 0 and 2 need proposals,
    # request 1 is in prefill
    proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    max_model_len = 20
    token_ids_cpu = np.zeros((3, max_model_len), dtype=np.int32)
    token_ids_cpu[0, :5] = [1, 2, 3, 1, 2]
    token_ids_cpu[1, :3] = [4, 5, 6]
    token_ids_cpu[2, :5] = [7, 8, 9, 7, 8]
    num_tokens_no_spec = np.array([5, 3, 5], dtype=np.int32)
    sampled_token_ids = [[2], [], [8]]  # Empty list for request 1 simulates prefill
    result = proposer.propose(
        num_speculative_tokens=2,
        sampled_token_ids=sampled_token_ids,
        num_tokens_no_spec=num_tokens_no_spec,
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result) == 3
    assert np.array_equal(result[0], [3, 1])
    assert len(result[1]) == 0
    assert np.array_equal(result[2], [9, 7])
    # Verify internal arrays written to correct indices
    assert proposer.valid_ngram_num_drafts[0] == 2
    assert proposer.valid_ngram_num_drafts[1] == 0
    assert proposer.valid_ngram_num_drafts[2] == 2
    assert np.array_equal(proposer.valid_ngram_draft[0, :2], [3, 1])
    assert np.array_equal(proposer.valid_ngram_draft[2, :2], [9, 7])

    # test if 0 threads available: can happen if TP size > CPU count
    ngram_proposer = get_ngram_proposer(min_n=2, max_n=2, k=2)
    ngram_proposer.num_numba_thread_available = 0
    # set max_model_len to 2 * threshold to ensure multithread is used
    num_tokens_threshold = ngram_proposer.num_tokens_threshold
    ngram_proposer.max_model_len = 2 * num_tokens_threshold
    # using multibatch test
    middle_integer = num_tokens_threshold // 2
    input_1 = [_ for _ in range(num_tokens_threshold)]
    input_1 += [middle_integer, middle_integer + 1]
    input_2 = [-1] * len(input_1)
    input_2[:3] = [4, 5, 6]
    token_ids_cpu = np.array([input_1, input_2])
    result = ngram_proposer.propose(
        num_speculative_tokens=2,
        sampled_token_ids=[[0], [1]],
        num_tokens_no_spec=np.array([len(input_1), 3]),
        token_ids_cpu=token_ids_cpu,
    )
    assert len(result[0]) == 2
    assert np.array_equal(result[0], np.array([middle_integer + 2, middle_integer + 3]))
    assert np.array_equal(result[1], np.array([]))
