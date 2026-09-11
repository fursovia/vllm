# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark editing, ordinary, and mixed traffic against a running chat server.

Run the same workload against V2, V1, and V1 ngram_gpu servers. With ngram_gpu,
run both with and without --prediction. Save individual responses for greedy
comparisons; stochastic runs compare quality/distributions rather than text.
"""

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import aiohttp
import numpy as np
from prometheus_client.parser import text_string_to_metric_families


def editing_cases() -> list[tuple[str, str]]:
    python_document = (
        "\n\n".join(
            f"def fetch_{name}(client, key):\n"
            f'    """Fetch {name.replace("_", " ")} by key."""\n'
            f'    response = client.get(f"/{name}/{{key}}", timeout=30)\n'
            "    response.raise_for_status()\n"
            "    return response.json()"
            for name in (
                "customers",
                "invoices",
                "payments",
                "subscriptions",
                "products",
                "prices",
                "credits",
                "refunds",
                "addresses",
                "contacts",
            )
        )
        + "\n"
    )
    runbook = (
        "# Northbridge release checklist\n\n"
        + "\n\n".join(
            f"## {index}. {title}\n\n"
            f"The Northbridge team owns this step. {instruction} "
            "Record the result in the release ticket before continuing. "
            "If the check fails, pause the release and contact the on-call engineer."
            for index, (title, instruction) in enumerate(
                (
                    (
                        "Prepare",
                        "Confirm the release tag and collect the approved changes.",
                    ),
                    (
                        "Back up",
                        "Create a database backup and verify that it is readable.",
                    ),
                    ("Deploy", "Deploy the tagged build to the staging environment."),
                    ("Verify", "Run the smoke tests and inspect the application logs."),
                    ("Release", "Roll out the build gradually and watch error rates."),
                    (
                        "Close",
                        "Confirm customer-facing checks and update the release notes.",
                    ),
                ),
                1,
            )
        )
        + "\n"
    )
    typescript = (
        "\n\n".join(
            f"export interface {name} {{\n"
            "  id: string;\n  displayName: string;\n  createdAt: string;\n"
            "  updatedAt: string;\n  active: boolean;\n}\n\n"
            f"export function describe{name}(item: {name}): string {{\n"
            "  return `${item.displayName} (${item.id})`;\n}"
            for name in (
                "Customer",
                "Project",
                "Workspace",
                "Team",
                "Account",
                "Product",
            )
        )
        + "\n"
    )
    return [
        ("Change the timeout from 30 to 45 in fetch_invoices only.", python_document),
        ("Replace Northbridge with Brookside throughout the document.", runbook),
        ("Rename displayName to label in every interface and function.", typescript),
    ]


async def metrics(session, base_url):
    async with session.get(base_url + "/metrics") as response:
        response.raise_for_status()
        body = await response.text()
    names = (
        "vllm:spec_decode_num_draft_tokens_total",
        "vllm:spec_decode_num_accepted_tokens_total",
    )
    result = dict.fromkeys(names, 0.0)
    for family in text_string_to_metric_families(body):
        for sample in family.samples:
            if sample.name in result:
                result[sample.name] += sample.value
    return result


async def request(session, args, index, traffic, temperature):
    editing = traffic == "editing" or (traffic == "mixed" and index % 2 == 0)
    instruction, document = editing_cases()[index % 3]
    messages = (
        [
            {
                "role": "system",
                "content": (
                    "Return only the complete revised document, preserving formatting. "
                    "Do not add explanations or code fences."
                ),
            },
            {"role": "user", "content": f"{instruction}\n\n{document}"},
        ]
        if editing
        else [
            {
                "role": "user",
                "content": (
                    "Explain one practical way to diagnose "
                    + (
                        "a slow database query",
                        "an intermittent HTTP timeout",
                        "a stale cache",
                    )[index % 3]
                    + ". Use six concise numbered steps."
                ),
            }
        ]
    )
    payload = {
        "model": args.model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.95,
        "top_k": 50,
        "seed": index,
        "max_completion_tokens": args.max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream": index % 2 == 0,
        "return_token_ids": True,
    }
    if editing and args.prediction:
        payload["prediction"] = {"type": "content", "content": document}
    if payload["stream"]:
        payload["stream_options"] = {"include_usage": True}
    result = await complete(session, args, payload)
    expected = (
        document.replace('/invoices/{key}", timeout=30', '/invoices/{key}", timeout=45')
        if index % 3 == 0
        else document.replace("Northbridge", "Brookside")
        if index % 3 == 1
        else document.replace("displayName", "label")
    )
    return {
        "index": index,
        "editing": editing,
        "edit_exact_match": result["text"].strip() == expected.strip()
        if editing
        else None,
        **result,
    }


async def complete(session, args, payload):
    start = time.perf_counter()
    text, token_ids, usage, reason = "", [], None, None
    async with session.post(
        args.base_url + "/v1/chat/completions", json=payload
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
        if payload["stream"]:
            async for raw in response.content:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if "error" in chunk:
                    raise RuntimeError(chunk)
                usage = chunk.get("usage") or usage
                for choice in chunk.get("choices", []):
                    text += choice["delta"].get("content") or ""
                    token_ids.extend(choice.get("token_ids") or [])
                    reason = choice.get("finish_reason") or reason
        else:
            result = await response.json()
            choice = result["choices"][0]
            text, token_ids = choice["message"]["content"], choice.get("token_ids")
            usage, reason = result["usage"], choice["finish_reason"]
    assert usage and usage["completion_tokens"] <= args.max_tokens
    assert text and reason in ("stop", "length"), (text, reason)
    return {
        "stream": payload["stream"],
        "latency_s": time.perf_counter() - start,
        "text": text,
        "token_ids": token_ids,
        "usage": usage,
        "finish_reason": reason,
    }


async def evaluate(session, args):
    from tests.evals.gsm8k.gsm8k_eval import get_answer_value, load_gsm8k_data

    _, test = load_gsm8k_data()
    runs = []
    for temperature in args.temperatures:
        semaphore = asyncio.Semaphore(8)

        async def one(index, item, temperature=temperature, semaphore=semaphore):
            payload = {
                "model": args.model,
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 50,
                "seed": index,
                "stream": False,
                "max_completion_tokens": args.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            item["question"]
                            + "\nSolve briefly and give the final numerical answer."
                        ),
                    }
                ],
            }
            if args.prediction:
                payload["prediction"] = {"type": "content", "content": item["answer"]}
            async with semaphore:
                record = await complete(session, args, payload)
            record["correct"] = get_answer_value(record["text"]) == get_answer_value(
                item["answer"]
            )
            record["index"] = index
            return record

        records = await asyncio.gather(
            *(
                one(index, item)
                for index, item in enumerate(test[: args.eval_questions])
            )
        )
        result = {
            "temperature": temperature,
            "questions": len(records),
            "accuracy": sum(record["correct"] for record in records) / len(records),
            "records": records,
        }
        runs.append(result)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "records"}
            ),
            flush=True,
        )
    return runs


async def run(args):
    output = {
        "args": vars(args),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runs": [],
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=300)
    ) as session:
        if args.eval_questions:
            output["evaluation"] = await evaluate(session, args)
            Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
            return
        for temperature in args.temperatures:
            for traffic in args.traffic:
                for concurrency in args.concurrencies:
                    await request(session, args, 0, traffic, temperature)
                    await asyncio.sleep(2)
                    before = await metrics(session, args.base_url)
                    semaphore = asyncio.Semaphore(concurrency)

                    async def one(
                        index,
                        semaphore=semaphore,
                        traffic=traffic,
                        temperature=temperature,
                    ):
                        async with semaphore:
                            return await request(
                                session, args, index, traffic, temperature
                            )

                    start = time.perf_counter()
                    records = await asyncio.gather(
                        *(one(i) for i in range(args.requests))
                    )
                    elapsed = time.perf_counter() - start
                    await asyncio.sleep(2)
                    after = await metrics(session, args.base_url)
                    delta = {name: after[name] - before[name] for name in before}
                    latencies = [record["latency_s"] for record in records]
                    run = {
                        "temperature": temperature,
                        "traffic": traffic,
                        "concurrency": concurrency,
                        "wall_s": elapsed,
                        "p50_s": float(np.percentile(latencies, 50)),
                        "p95_s": float(np.percentile(latencies, 95)),
                        "requests_per_s": len(records) / elapsed,
                        "output_tokens_per_s": sum(
                            record["usage"]["completion_tokens"] for record in records
                        )
                        / elapsed,
                        "speculation": delta,
                        "records": records,
                    }
                    output["runs"].append(run)
                    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
                    print(
                        json.dumps(
                            {
                                key: value
                                for key, value in run.items()
                                if key != "records"
                            }
                        ),
                        flush=True,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8125")
    parser.add_argument("--model", default="google/gemma-4-E2B-it")
    parser.add_argument("--prediction", action="store_true")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--eval-questions", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0, 0.7, 1])
    parser.add_argument(
        "--traffic",
        nargs="+",
        choices=["editing", "ordinary", "mixed"],
        default=["editing", "ordinary", "mixed"],
    )
    asyncio.run(run(parser.parse_args()))
