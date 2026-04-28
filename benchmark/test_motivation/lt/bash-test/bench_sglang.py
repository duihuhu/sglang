import argparse
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "C_nvml_for_energy"))
from nvml_energy import NvmlEnergy


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


def completion_tokens_from_generate_response(resp_json: Any) -> tuple[int, int]:
    """Parse non-streaming /generate JSON: single dict or batch list. Returns (sum_tokens, n_subresponses)."""
    if isinstance(resp_json, list):
        total = 0
        for item in resp_json:
            if not isinstance(item, dict):
                raise RuntimeError(f"Unexpected batch response element type: {type(item)}")
            ct = extract_completion_tokens(item)
            if ct is None:
                raise RuntimeError(
                    f"Cannot extract completion_tokens from response item: {item!r}"
                )
            total += ct
        return total, len(resp_json)
    if not isinstance(resp_json, dict):
        raise RuntimeError(f"Unexpected /generate response type: {type(resp_json)}")
    ct = extract_completion_tokens(resp_json)
    if ct is None:
        raise RuntimeError(
            f"Cannot extract completion_tokens from response: {resp_json!r}"
        )
    return ct, 1


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
        help=(
            "Sequences per measurement round. For batch_size>1, uses one HTTP POST "
            "with input_ids as list-of-lists so the server runs a single batched prefill."
        ),
    )
    parser.add_argument("--input_len", type=int, default=32, help="Input token length (pre-tokenized input_ids)")
    parser.add_argument("--output_len", type=int, default=32, help="max_new_tokens per request")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--sm_clock", type=int, default=1200, help="SM clock frequency to lock (MHz)")
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

    with NvmlEnergy() as nv:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            gpu_ids = [int(x) for x in cvd.split(",") if x.strip()]
        else:
            gpu_ids = list(range(nv.device_count()))
        for g in gpu_ids:
            nv.lock_sm_clock(g, args.sm_clock)
        print(f"GPU {gpu_ids} SM 已锁定到 {args.sm_clock} MHz")

        try:
            generate_url = args.server_url.rstrip("/") + "/generate"
            server_info_url = args.server_url.rstrip("/") + "/get_server_info"

            sampling_params = {
                "temperature": args.temperature,
                "max_new_tokens": args.output_len,
                "sampling_seed": args.seed,
            }
            if args.ignore_eos:
                sampling_params["ignore_eos"] = True

            prompts: List[List[int]] = [
                [random.randint(0, args.vocab_max) for _ in range(args.input_len)]
                for _ in range(args.num_seqs)
            ]

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

                if len(batch_indexes) > 1:
                    # Concurrent single-seq POSTs are often scheduled as separate forwards;
                    # PD bench window matching expects one extend with seq_lens_sum=input_len*batch_size.
                    payload = {
                        "input_ids": [prompts[i] for i in batch_indexes],
                        "sampling_params": copy.deepcopy(sampling_params),
                    }
                    with requests.Session() as worker_session:
                        resp_json = post_json(
                            worker_session, generate_url, payload, timeout_s=args.timeout_s
                        )
                    batch_tokens, batch_completed = completion_tokens_from_generate_response(
                        resp_json
                    )
                    print(
                        f"end batched POST [{batch_start}, {batch_end}): "
                        f"subresponses={batch_completed}, completion_tokens_sum={batch_tokens}"
                    )
                else:
                    req_idx = batch_indexes[0]
                    payload = {
                        "input_ids": prompts[req_idx],
                        "sampling_params": copy.deepcopy(sampling_params),
                    }
                    with requests.Session() as worker_session:
                        resp_json = post_json(
                            worker_session, generate_url, payload, timeout_s=args.timeout_s
                        )
                    batch_tokens, batch_completed = completion_tokens_from_generate_response(
                        resp_json
                    )
                    print(f"end {req_idx} (completion_tokens={batch_tokens})")

                print(f"end batch [{batch_start}, {batch_end})")
                garbage_collection()
                return batch_tokens, batch_completed

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
            for g in gpu_ids:
                nv.unlock_sm_clock(g)
            print(f"GPU {gpu_ids} SM 锁频已清除")


if __name__ == "__main__":
    main()

