# Prediction API implementation plan

## Goal and scope

Accept OpenAI-style `prediction: {"type": "content", "content": "..."}` on
`/v1/chat/completions` and accelerate generation by extending `ngram_gpu`.
Preserve asynchronous scheduling and reuse existing speculative verification,
rejection sampling, KV-cache management, and streaming.

Start with text-only requests, string prediction content, and `n=1`. Support
both greedy (`temperature=0`) and nonzero-temperature sampling from the start.
Validate on `google/gemma-4-E2B-it` with thinking disabled at temperatures
`0`, `0.7`, and `1.0`; keep the implementation model-independent. Use the
existing V1 model runner, which supports asynchronous scheduling with `ngram_gpu`.
Keep the first implementation focused on prediction transport, GPU state, and
corpus matching. Defer a V2 port unless measurements or upstream review justify
the extra work.

Configure `num_speculative_tokens=8` server-wide initially. This caps draft
tokens per verification step, not the total prediction length. Reuse existing
n-gram lookup settings. Defer content-part arrays, per-request block sizes,
tools, multimodal requests, structured outputs, and additional speculation methods.

## Implementation

1. **Accept and transport the prediction.**
   - Add the typed field in `vllm/entrypoints/openai/chat_completion/protocol.py`.
   - In `chat_completion/serving.py`, validate the supported request/configuration,
     tokenize once with the target tokenizer and `add_special_tokens=False`,
     and enforce a bounded prediction length. Preserve whitespace.
   - Add optional `prediction_token_ids` to `vllm/sampling_params.py`; existing
     request transport already carries sampling parameters to the worker.
   - Require V1 with `ngram_gpu` for nonempty predictions and return a clear API
     error for unsupported combinations, including `use_beam_search=True` even
     with `n=1`. Missing or empty predictions preserve existing behavior.
     Prediction text is separate from the prompt; editing instructions and the
     original document belong in messages.

2. **Keep prediction state on the GPU.**
   - Extend `vllm/v1/worker/gpu_model_runner.py` using its existing n-gram
     request/batch lifecycle hooks.
   - Upload immutable prediction tokens and lengths when requests enter the
     worker. Keep buffers aligned through batch reordering, preemption/resumption,
     completion, cancellation, and slot reuse.
   - Continue using device-side accepted-token counts and existing asynchronous
     draft transfer. Do not introduce per-step CPU reads of generated tokens.

3. **Extend the GPU matcher.**
   - In `vllm/v1/spec_decode/ngram_proposer_gpu.py`, separate the generated suffix
     used for matching from the corpus searched for candidate continuations.
   - For prediction requests, search the prediction for the longest matching
     output suffix and propose the following tokens. Use existing n-gram
     behavior for other requests in the same batch.
   - For prediction requests, match only generated output, without crossing the
     prompt boundary. Wait for at least `prompt_lookup_min` output tokens before
     drafting. Defaults require a five-token match, so predictions of five tokens
     or fewer cannot accelerate generation; keep this limitation explicit.
   - Recompute the match each step so speculation can resume after edits.
     Use deterministic tie-breaking; no cursor or edit-distance algorithm.
   - Respect prediction boundaries and context/output limits. Mask padding and
     no-match rows, returning existing draft IDs and valid-count formats.
     Exhausting a prediction must not end generation.
   - Update warmup inputs for the extended matcher; retain existing verification.
   - Pass deterministic draft tokens with `draft_probs=None` to the existing
     rejection sampler. Reuse its greedy and random sampling paths, including
     temperature and top-k/top-p handling; no new sampling algorithm is needed.

## Server protection (mandatory)

Keeping existing services and processes on `b200` undisturbed is a hard
constraint. GPU isolation alone does not isolate shared CPU, RAM, disk, or I/O.

- Leave system CUDA, NVIDIA drivers, system packages, other Python environments,
  and service configuration unchanged. Do not restart services or the host.
  Install task dependencies only in a dedicated `.venv`; CUDA Python packages
  there must not replace system CUDA or affect other environments.
- Before switching branches, pulling, or changing dependencies, use read-only
  checks to determine whether running services use the checkout or its `.venv`.
  If either is in use, use a separate worktree and dedicated `.venv`. If usage
  cannot be established, stop before changing them.
- Use physical GPU 5 exclusively, as specified below. Do not inspect, initialize,
  reset, reconfigure, or stop jobs on any other GPU. Do not displace an existing
  job on GPU 5.
- Before installations, builds, model loading, or benchmarks, check shared CPU,
  RAM, disk space, and I/O headroom. Apply task-scoped resource limits and bounded
  build/thread concurrency; run model configurations sequentially. Record the
  limits used and monitor resource pressure during testing.
- If sufficient isolation or headroom cannot be established, stop testing.
  If resource pressure threatens existing workloads, stop only this task's
  processes. Never free resources by killing unrelated jobs or deleting their
  files or caches.

## Development and test workflow

1. Implement all code, tests, and benchmark scripts locally in
   `/Users/fursov/Documents/vllm` on a `features-` branch. Run applicable local
   lint and CPU checks, then commit and push the branch.
2. Connect to `b200` and inspect `/home/fursov/vllm`, following the server
   protection rules above. Use a separate worktree if the checkout or environment
   is in use. Preserve unrelated work, check out the same branch, and pull with
   `--ff-only`. Record the test checkout and environment paths, and verify that
   the server HEAD matches the pushed commit before testing.
3. **Use physical GPU index 5 exclusively. Other GPUs must not be touched.**
   Resolve its UUID with
   `nvidia-smi -i 5 --query-gpu=uuid --format=csv,noheader` and set
   `CUDA_VISIBLE_DEVICES` to that UUID for every GPU test, model server,
   benchmark, and child process. GPU 5 then appears inside the process as
   `cuda:0`. Keep tensor, pipeline, and data parallelism at 1; run model
   configurations sequentially. Do not inspect, initialize, or modify other
   GPUs or disturb their jobs. If GPU 5 is unavailable, stop GPU testing;
   never fall back to another GPU.
4. Test `google/gemma-4-E2B-it` on that isolated GPU, following the checks below.
   Use `uv` for environment/dependency management and `.venv/bin/python` for
   Python commands. Record the tested commit, GPU UUID, commands, and results.
5. For every fix, edit locally, rerun applicable local checks, commit/push,
   then pull and retest on `b200`. Never edit implementation or test scripts
   directly on the server.

## Validation and completion criteria

- Extend nearby API and n-gram tests: valid/invalid requests (including beam
  search rejection), initial matching and short predictions, exact drafts,
  resumption after insertions/deletions/replacements, repeated matches, empty or
  unrelated drafts, and stopping limits.
- Exercise asynchronous batches mixing different predictions and ordinary
  requests and mixing greedy/nonzero temperatures, including batch movement
  and slot reuse; verify streaming output.
- Compare Gemma greedy outputs against V1 generation without speculation and
  investigate differences. For nonzero temperatures, reuse/extend existing
  sampler distribution tests with deterministic drafts and run model evaluation
  and end-to-end sampling checks. Check preservation of the target distribution,
  not identical text for the same seed; speculation changes random-number use.
- Benchmark draft sizes 4/8/16 on realistic editing requests at low and moderate
  concurrency at temperatures `0`, `0.7`, and `1.0`. The primary baseline is the
  default V2 server without speculation; verify the selected runner. Also compare
  against V1 without speculation and ordinary V1 `ngram_gpu` to isolate the
  effects of speculation and prediction. Pin every V1 configuration with
  `VLLM_USE_V2_MODEL_RUNNER=0`; keep workloads and sampling settings consistent.
- Record end-to-end p50/p95 latency (including prediction tokenization and
  upload), throughput, acceptance, and added GPU memory. Include ordinary-only
  and mixed traffic to check regressions against the baselines. Require meaningful
  editing latency gains over default V2 without materially slowing ordinary
  requests. Report gains by temperature: nonzero temperatures can reduce
  acceptance and speedup. Reassess the approach if it does not demonstrate benefit.
- Document the API, supported scope, launch settings, and measured results.

## Existing work

[PR 23450](https://github.com/vllm-project/vllm/pull/23450) and
[PR 24568](https://github.com/vllm-project/vllm/pull/24568) were closed without
merging. Recheck duplicates before proposing a PR. Use existing speculative
acceptance metrics initially; avoid duplicating the open
[usage-counter PR 51778](https://github.com/vllm-project/vllm/pull/51778).

Reference: [OpenAI Predicted Outputs](https://developers.openai.com/api/docs/guides/predicted-outputs).
This plan proposes an implementation, not a reproduction of OpenAI's internals.
