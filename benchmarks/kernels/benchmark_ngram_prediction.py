# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare compiled ordinary and prediction n-gram matching with CUPTI timing."""

import argparse
import json
import time
from functools import partial

import numpy as np
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.config import (
    ModelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU


def benchmark(args):
    for length in args.lengths:
        config = VllmConfig(
            model_config=ModelConfig(model="facebook/opt-125m", max_model_len=length),
            scheduler_config=SchedulerConfig(
                max_num_seqs=max(args.batches),
                max_model_len=length,
                is_encoder_decoder=False,
            ),
            speculative_config=SpeculativeConfig(
                method="ngram_gpu",
                num_speculative_tokens=8,
                prompt_lookup_min=5,
                prompt_lookup_max=5,
            ),
        )
        start = time.perf_counter()
        with set_current_vllm_config(config):
            proposer = NgramProposerGPU(config, torch.device("cuda:0"))
        torch.accelerator.synchronize()
        print(
            json.dumps(
                {"length": length, "compile_and_warmup_s": time.perf_counter() - start}
            )
        )
        for batch in args.batches:
            tokens = torch.arange(length, dtype=torch.int32, device="cuda").repeat(
                batch, 1
            )
            tokens[:, -5:] = torch.arange(5, dtype=torch.int32, device="cuda")
            seq_lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")
            mask = torch.ones(batch, dtype=torch.bool, device="cuda")
            corpus = torch.arange(length, dtype=torch.int32, device="cuda").repeat(
                batch, 1
            )
            prompt_lengths = torch.zeros_like(seq_lengths)
            # No output limit for the kernel-only comparison.
            token_limits = seq_lengths + 8
            with set_forward_context(None, proposer.vllm_config):
                for mode in ("ordinary", "prediction"):
                    kernel = (
                        proposer.kernel
                        if mode == "ordinary"
                        else proposer.prediction_kernel
                    )
                    inputs = (seq_lengths, tokens, mask)
                    if mode == "prediction":
                        inputs += (corpus, seq_lengths, prompt_lengths, token_limits)
                    expected = torch.arange(5, 13, device="cuda").repeat(batch, 1)
                    actual, counts = kernel(*inputs)
                    torch.testing.assert_close(actual.long(), expected)
                    assert counts.tolist() == [8] * batch
                    times = bench_gpu_time_with_cupti(
                        partial(kernel, *inputs),
                        use_cuda_graph=True,
                        cold_l2_cache=True,
                    )
                    print(
                        json.dumps(
                            {
                                "length": length,
                                "batch": batch,
                                "mode": mode,
                                "median_us": float(np.median(times) * 1000),
                                "effective_read_gb_s": (
                                    4
                                    * batch
                                    * length
                                    * (1 if mode == "ordinary" else 2)
                                    / (float(np.median(times)) * 1e6)
                                ),
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 4096, 16384])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    benchmark(parser.parse_args())
