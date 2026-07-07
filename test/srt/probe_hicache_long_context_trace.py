import argparse
import json
import os
import random
import shutil
import tempfile
import time
from urllib.parse import urlparse

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    flush_cache_with_retry,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready


def get_server_info(base_url):
    response = requests.get(f"{base_url}/server_info", timeout=30)
    response.raise_for_status()
    internal_states = response.json().get("internal_states", [])
    if not internal_states:
        return {}
    return internal_states[0].get("hicache_extent_debug", {})


def wait_for_hicache_idle(base_url, timeout=120.0):
    deadline = time.time() + timeout
    last_debug = None
    while time.time() < deadline:
        last_debug = get_server_info(base_url)
        ongoing = [
            last_debug.get("ongoing_write_through", 0),
            last_debug.get("ongoing_load_back", 0),
            last_debug.get("ongoing_prefetch", 0),
            last_debug.get("ongoing_backup", 0),
        ]
        if all(value == 0 for value in ongoing):
            return last_debug
        time.sleep(1)
    raise TimeoutError(f"HiCache async operations did not drain: {last_debug}")


def gen_prompt(tokenizer, token_num, seed):
    rng = random.Random(seed)
    vocab_ids = list(tokenizer.get_vocab().values())
    selected_tokens = rng.choices(vocab_ids, k=token_num)
    return tokenizer.decode(selected_tokens)


def send_generate(base_url, prompt, max_tokens):
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_tokens,
                "ignore_eos": True,
            },
        },
        timeout=600,
    )
    latency = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    cached_tokens = int(body.get("meta_info", {}).get("cached_tokens", 0))
    return {"latency_s": latency, "cached_tokens": cached_tokens}


def summarize_records(records):
    groups = [group for record in records for group in record.get("groups", [])]
    runs = [group.get("avg_run_pages", 0.0) for group in groups]
    pages = [group.get("pages", 0) for group in groups]
    bytes_ = [group.get("bytes", 0) for group in groups]
    return {
        "records": len(records),
        "groups": len(groups),
        "avg_group_pages": (sum(pages) / len(pages)) if pages else 0.0,
        "avg_group_bytes": (sum(bytes_) / len(bytes_)) if bytes_ else 0.0,
        "min_group_pages": min(pages) if pages else 0,
        "max_group_pages": max(pages) if pages else 0,
        "avg_group_run_pages": (sum(runs) / len(runs)) if runs else 0.0,
        "one_page_run_groups": sum(1 for value in runs if value == 1.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/data/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--base-url", default=DEFAULT_URL_FOR_TEST)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--question-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--grow-tokens", type=int, default=8192)
    parser.add_argument("--leave-old-tokens", type=int, default=1024)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    parsed_url = urlparse(args.base_url)
    temp_dir = tempfile.mkdtemp()
    process = None
    try:
        tokenizer = get_tokenizer(args.model_path)
        prefix = gen_prompt(tokenizer, args.prefix_tokens, args.seed)
        questions = [
            gen_prompt(tokenizer, args.question_tokens, args.seed + 1000 + i)
            for i in range(args.requests)
        ]
        prompts = [prefix + "\n\nQuestion:\n" + question for question in questions]

        server_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            str(args.page_size),
            "--enable-cache-report",
            "--hicache-storage-prefetch-policy",
            "wait_complete",
            "--hicache-storage-backend",
            "file",
            "--hicache-storage-backend-extra-config",
            json.dumps({"hicache_storage_pass_prefix_keys": True}),
            "--hicache-mem-layout",
            "page_first_direct",
            "--hicache-io-backend",
            "direct",
        ]
        env = {
            **os.environ,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": temp_dir,
            "SGLANG_TEST_HICACHE_GROW_EXTENT_TOKENS": str(args.grow_tokens),
            "SGLANG_TEST_HICACHE_RESERVE_OLD_EXTENT_LEAVE_TOKENS": str(
                args.leave_old_tokens
            ),
            "SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS": "1",
        }

        process = popen_launch_server(
            args.model_path,
            args.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=env,
        )
        wait_for_http_ready(
            url=f"{args.base_url}/health", timeout=120, process=process
        )

        before = get_server_info(args.base_url)
        first_pass = [
            send_generate(args.base_url, prompt, args.max_new_tokens)
            for prompt in prompts
        ]
        after_first_pass = wait_for_hicache_idle(args.base_url)
        if not flush_cache_with_retry(args.base_url):
            raise RuntimeError("Cache flush failed")
        second_pass = [
            send_generate(args.base_url, prompt, args.max_new_tokens)
            for prompt in prompts
        ]
        after_second_pass = wait_for_hicache_idle(args.base_url)

        records = after_second_pass.get("test_extent_transfer_records", [])
        result = {
            "config": {
                "model_path": args.model_path,
                "host": parsed_url.hostname,
                "port": parsed_url.port,
                "prefix_tokens": args.prefix_tokens,
                "question_tokens": args.question_tokens,
                "requests": args.requests,
                "max_new_tokens": args.max_new_tokens,
                "page_size": args.page_size,
                "grow_tokens": args.grow_tokens,
                "leave_old_tokens": args.leave_old_tokens,
            },
            "first_pass": first_pass,
            "second_pass": second_pass,
            "extent_before": before,
            "extent_after_first_pass": after_first_pass,
            "extent_after_second_pass": after_second_pass,
            "transfer_summary": after_second_pass.get(
                "test_extent_transfer_summary", {}
            ),
            "raw_record_summary": summarize_records(records),
            "transfer_records": records,
        }

        with open(args.output, "w") as fout:
            json.dump(result, fout, indent=2)
        print(json.dumps(result["transfer_summary"], indent=2))
        print(json.dumps(result["raw_record_summary"], indent=2))
        print(f"Wrote {args.output}")
    finally:
        if process is not None:
            kill_process_tree(process.pid)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
