import argparse
import concurrent.futures
import copy
import os
import random
import sys
import time
import gc
import torch
from typing import Any, Dict, List, Optional

import requests

sys.path.append(os.path.dirname(__file__))
from utils import reset_clocks, set_all_visible_gpus_clock


def garbage_collection():
    gc.collect()
    torch.cuda.empty_cache()

def post_json(
    session: requests.Session,
    url: str,
    payload: Dict[str, Any],
    timeout_s: int,
    expect_json: bool = True,
):
    r = session.post(url, json=payload, timeout=timeout_s)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        body = ""
        if r is not None:
            body = (r.text or "").strip()
            if len(body) > 800:
                body = body[:800] + "...(truncated)"
        raise requests.HTTPError(
            f"{e}. response_body={body}",
            response=r,
        ) from e
    if not expect_json:
        return r.text
    if not r.content:
        return {}
    try:
        return r.json()
    except ValueError:
        # Some endpoints (e.g. /start_profile, /stop_profile) return plain text.
        return {"text": r.text}


def extract_completion_tokens(resp_json: Dict[str, Any]) -> Optional[int]:
    """
    Non-streaming /generate typically returns:
      { "output_ids" or "text", "meta_info": { "completion_tokens": int, ... } }
    """
    meta = resp_json.get("meta_info") or {}
    ct = meta.get("completion_tokens")
    if isinstance(ct, int):
        return ct
    # Fallback: if output_ids is returned (token-id generation)
    out_ids = resp_json.get("output_ids")
    if isinstance(out_ids, list):
        return len(out_ids)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang with batched /generate requests"
    )
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:30000", help="SGLang HTTP base URL")
    parser.add_argument(
        "--num_seqs",
        type=int,
        default=None,
        help="Total number of requests to send. If omitted, defaults to --batch_size.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,                                                                                                                                         
        default=1,
        help="Number of /generate requests to run concurrently in one batch.",
    )
    parser.add_argument("--input_len", type=int, default=32, help="Input token length (pre-tokenized input_ids)")
    parser.add_argument("--output_len", type=int, default=32, help="max_new_tokens per request")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--mem_clock", type=int, default=1593, help="GPU memory clock (MHz)")                                                                                                                            
    parser.add_argument("--gpu_clock", type=int, default=1200, help="GPU graphics/memory clock (MHz)")
    parser.add_argument("--ignore_eos", action="store_true", help="Set sampling_params.ignore_eos=true")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generating input_ids")                                                            
    parser.add_argument("--vocab_max", type=int, default=10000, help="Upper bound for random input_ids (exclusive)")
    parser.add_argument("--timeout_s", type=int, default=600, help="HTTP timeout seconds per request")


    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be >= 1")
    if args.num_seqs is None:
        args.num_seqs = args.batch_size
    if args.num_seqs <= 0:
        raise ValueError("--num_seqs must be >= 1")

    random.seed(args.seed)
    session = requests.Session()

    # Tune GPU clocks before generating to reduce run-to-run variance.
    set_all_visible_gpus_clock(mem_clock=args.mem_clock, graphics_clock=args.gpu_clock)

    try:
        generate_url = args.server_url.rstrip("/") + "/generate"
        server_info_url = args.server_url.rstrip("/") + "/get_server_info"

        sampling_params = {
            "temperature": args.temperature,
            "max_new_tokens": args.output_len,
            # Make generation reproducible even when temperature > 0.
            # Note: this seed affects sampling, not prompt token generation.
            "sampling_seed": args.seed,
        }
        if args.ignore_eos:
            sampling_params["ignore_eos"] = True

        prompts: List[List[int]] = [
            [random.randint(0, args.vocab_max) for _ in range(args.input_len)]
            for _ in range(args.num_seqs)
        ]

        # Pre-check context length to avoid opaque 400 errors from /generate.
        try:
            server_info = session.get(server_info_url, timeout=args.timeout_s).json()
            context_len = server_info.get("context_len")
            if isinstance(context_len, int):
                if args.input_len >= context_len:
                    raise ValueError(
                        f"input_len={args.input_len} is >= context_len={context_len}. "
                        "Reduce --input_len."
                    )
                if args.input_len + args.output_len >= context_len:
                    raise ValueError(
                        f"input_len + output_len = {args.input_len + args.output_len} "
                        f"is >= context_len={context_len}. Reduce --input_len or --output_len."
                    )
        except requests.RequestException:
            # Best-effort pre-check: continue if /get_server_info is unavailable.
            pass

        print("start to test-------------------")
        print(f"num_seqs={args.num_seqs}, batch_size={args.batch_size}")
        t0 = time.time()
        total_tokens = 0
        completed = 0

        def _run_batch(batch_start: int) -> tuple[int, int]:
            batch_end = min(batch_start + args.batch_size, args.num_seqs)
            batch_indexes = list(range(batch_start, batch_end))
            print(f"start batch [{batch_start}, {batch_end})")
            batch_tokens = 0
            batch_completed = 0

            def _run_one(req_idx: int) -> int:
                payload = {
                    "input_ids": prompts[req_idx],
                    "sampling_params": copy.deepcopy(sampling_params),
                }
                # One session per worker to avoid thread-safety issues.
                with requests.Session() as worker_session:
                    resp_json = post_json(
                        worker_session, generate_url, payload, timeout_s=args.timeout_s
                    )
                ct = extract_completion_tokens(resp_json)
                if ct is None:
                    raise RuntimeError(
                        f"Cannot extract completion_tokens from response: {resp_json}"
                    )
                return ct

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(batch_indexes)
            ) as executor:
                future_to_idx = {
                    executor.submit(_run_one, req_idx): req_idx for req_idx in batch_indexes
                }
                for fut in concurrent.futures.as_completed(future_to_idx):
                    req_idx = future_to_idx[fut]
                    ct = fut.result()
                    batch_tokens += ct
                    batch_completed += 1
                    print(f"end {req_idx} (completion_tokens={ct})")
            print(f"end batch [{batch_start}, {batch_end})")
            garbage_collection()
            return batch_tokens, batch_completed

        # Single measured pass only.
        for batch_start in range(0, args.num_seqs, args.batch_size):
            tokens, done = _run_batch(batch_start)
            total_tokens += tokens
            completed += done

        t = time.time() - t0

        throughput = total_tokens / t if t > 0 else 0.0
        print(f"Total requests: {completed}")
        print(
            f"Total tokens: {total_tokens} tok, Time: {t:.2f} s, Throughput: {throughput:.2f} tok/s"
        )
        print("Done.")
    finally:
        # Always restore clocks.
        reset_clocks()


if __name__ == "__main__":
    main()

