import argparse
import ast
import json
import os
import shutil
import tempfile
import time

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, popen_launch_server
from sglang.utils import wait_for_http_ready


def get_extent_debug(base_url):
    response = requests.get(f"{base_url}/server_info", timeout=30)
    response.raise_for_status()
    states = response.json().get("internal_states", [])
    if not states:
        return {}
    return states[0].get("hicache_extent_debug", {})


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_loogle_prompts(dataset_path, num_docs):
    prompts = []
    with open(dataset_path) as fin:
        for line in fin:
            if len(prompts) >= num_docs:
                break
            row = json.loads(line)
            qa_pairs = ast.literal_eval(row["qa_pairs"])
            prompts.append(
                [
                    {
                        "doc_id": row.get("doc_id"),
                        "title": row.get("title"),
                        "question_index": index,
                        "text": "Input: " + row["input"] + " Question: " + qa["Q"],
                    }
                    for index, qa in enumerate(qa_pairs)
                ]
            )
    return prompts


def load_loogle_prompt_groups(dataset_path, start_doc, num_docs, questions_per_doc=0):
    groups = load_loogle_prompts(dataset_path, start_doc + num_docs)
    selected = groups[start_doc : start_doc + num_docs]
    if questions_per_doc > 0:
        selected = [group[:questions_per_doc] for group in selected]
    return selected


def flatten_prompt_groups(prompt_groups):
    return [prompt for group in prompt_groups for prompt in group]


def send_generate(base_url, prompt, max_new_tokens):
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt["text"],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
                "ignore_eos": True,
            },
        },
        timeout=1200,
    )
    latency = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    return {
        "doc_id": prompt["doc_id"],
        "title": prompt["title"],
        "question_index": prompt["question_index"],
        "latency_s": latency,
        "cached_tokens": int(body.get("meta_info", {}).get("cached_tokens", 0)),
    }


def run_prompt_pass(base_url, prompts, max_new_tokens):
    results = []
    for prompt in prompts:
        results.append(send_generate(base_url, prompt, max_new_tokens))
    return results


def wait_for_hicache_idle(base_url, timeout=180.0):
    deadline = time.time() + timeout
    last_debug = None
    while time.time() < deadline:
        last_debug = get_extent_debug(base_url)
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


def flush_cache(base_url, retries=20):
    last_text = ""
    for _ in range(retries):
        response = requests.post(f"{base_url}/flush_cache", timeout=30)
        if response.status_code == 200:
            return True
        last_text = response.text
        time.sleep(1)
    raise RuntimeError(f"Cache flush failed after {retries} retries: {last_text}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-path", default="/mnt/data/models/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--dataset-path",
        default="/mnt/data/dpser/datasets/loogle/longdep_qa_sglang_qa_pairs.jsonl",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:21000")
    parser.add_argument("--mem-layout", required=True)
    parser.add_argument("--io-backend", required=True)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=1)
    parser.add_argument(
        "--storage-backend",
        choices=["none", "file"],
        default="none",
        help="Use none for host-pinned HiCache pressure tests; file for L3 flush/replay tests.",
    )
    parser.add_argument(
        "--replay-mode",
        choices=["pressure", "flush", "immediate"],
        default="pressure",
        help="pressure evicts GPU KV without clearing host; flush requires file storage.",
    )
    parser.add_argument("--evict-docs", type=int, default=2)
    parser.add_argument("--evict-questions-per-doc", type=int, default=1)
    parser.add_argument("--fixed-output-len", type=int, default=16)
    parser.add_argument("--request-rate", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--grow-tokens", type=int, default=0)
    parser.add_argument("--leave-old-tokens", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    temp_dir = tempfile.mkdtemp(prefix=f"hicache-{args.tag}-")
    process = None
    try:
        server_args = [
            "--enable-hierarchical-cache",
            "--mem-fraction-static",
            "0.6",
            "--hicache-ratio",
            "1.2",
            "--page-size",
            str(args.page_size),
            "--enable-cache-report",
            "--hicache-mem-layout",
            args.mem_layout,
            "--hicache-io-backend",
            args.io_backend,
        ]
        if args.storage_backend != "none":
            server_args.extend(
                [
                    "--hicache-storage-prefetch-policy",
                    "wait_complete",
                    "--hicache-storage-backend",
                    args.storage_backend,
                    "--hicache-storage-backend-extra-config",
                    json.dumps({"hicache_storage_pass_prefix_keys": True}),
                ]
            )
        if args.replay_mode == "flush" and args.storage_backend == "none":
            raise ValueError("--replay-mode flush requires --storage-backend file")
        env = {
            **os.environ,
            "SGLANG_ENABLE_DETERMINISTIC_INFERENCE": "1",
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": temp_dir,
            "SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS": "1",
        }
        if args.grow_tokens > 0:
            env["SGLANG_TEST_HICACHE_GROW_EXTENT_TOKENS"] = str(args.grow_tokens)
        if args.leave_old_tokens > 0:
            env["SGLANG_TEST_HICACHE_RESERVE_OLD_EXTENT"] = "1"
            env["SGLANG_TEST_HICACHE_RESERVE_OLD_EXTENT_LEAVE_TOKENS"] = str(
                args.leave_old_tokens
            )

        process = popen_launch_server(
            args.model_path,
            args.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=server_args,
            env=env,
        )
        wait_for_http_ready(
            url=f"{args.base_url}/health",
            timeout=120,
            process=process,
        )
        prompt_groups = load_loogle_prompts(args.dataset_path, args.num_prompts)
        prompts = flatten_prompt_groups(prompt_groups)
        before_debug = get_extent_debug(args.base_url)

        started = time.perf_counter()
        populate_results = run_prompt_pass(
            args.base_url, prompts, args.fixed_output_len
        )
        populate_elapsed = time.perf_counter() - started
        after_populate_debug = wait_for_hicache_idle(args.base_url)

        evict_results = []
        evict_elapsed = 0.0
        after_evict_debug = None
        after_flush_debug = None
        if args.replay_mode == "flush":
            flush_cache(args.base_url)
            after_flush_debug = get_extent_debug(args.base_url)
        elif args.replay_mode == "pressure":
            evict_groups = load_loogle_prompt_groups(
                args.dataset_path,
                start_doc=args.num_prompts,
                num_docs=args.evict_docs,
                questions_per_doc=args.evict_questions_per_doc,
            )
            evict_prompts = flatten_prompt_groups(evict_groups)
            started = time.perf_counter()
            evict_results = run_prompt_pass(
                args.base_url, evict_prompts, args.fixed_output_len
            )
            evict_elapsed = time.perf_counter() - started
            after_evict_debug = wait_for_hicache_idle(args.base_url)

        started = time.perf_counter()
        replay_results = run_prompt_pass(args.base_url, prompts, args.fixed_output_len)
        replay_elapsed = time.perf_counter() - started
        after_replay_debug = wait_for_hicache_idle(args.base_url)

        result = {
            "tag": args.tag,
            "config": vars(args),
            "server_args": server_args,
            "prompt_group_count": len(prompt_groups),
            "request_count": len(prompts),
            "populate_elapsed_s": populate_elapsed,
            "evict_elapsed_s": evict_elapsed,
            "replay_elapsed_s": replay_elapsed,
            "populate_results": populate_results,
            "evict_results": evict_results,
            "replay_results": replay_results,
            "extent_before": before_debug,
            "extent_after_populate": after_populate_debug,
            "extent_after_evict": after_evict_debug,
            "extent_after_flush": after_flush_debug,
            "extent_after_replay": after_replay_debug,
        }
        with open(args.output, "w") as fout:
            json.dump(result, fout, indent=2)
        print(json.dumps(result, indent=2)[-12000:])
    finally:
        if process is not None:
            kill_process_tree(process.pid)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
