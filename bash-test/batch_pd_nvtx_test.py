#!/workspace/env/sglang-main/bin/python
import argparse
import contextlib
import csv
import glob
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from queue import Queue
import shutil
import uuid
from typing import Dict, List, Optional, Sequence, Tuple

# Repo root (for PYTHONPATH when nsys cwd is per-run directory).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_CACHE_ROOT = "/mnt/nvme1/lt/cache"
def _apply_json_config(args: argparse.Namespace) -> Optional[dict]:
    cfg_path = str(getattr(args, "config", "") or "").strip()
    if not cfg_path:
        return None
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError("--config JSON must be an object")
    list_to_csv_keys = {
        "tp_list",
        "input_lens",
        "output_lens",
        "gpu_clocks",
        "batch_size",
        "gpus",
    }
    for k, v in cfg.items():
        if hasattr(args, k):
            if k in list_to_csv_keys and isinstance(v, list):
                v = ",".join(str(x) for x in v)
            setattr(args, k, v)
    return cfg


def _to_int_list(v) -> List[int]:
    if isinstance(v, list):
        return [int(x) for x in v]
    return [int(v)]



@dataclass(frozen=True)
class JobSpec:
    tp: int
    input_len: int
    gpu_clock: int
    output_len_max: int
    batch_size: int


def _run_cmd(
    cmd: List[str],
    env: Dict[str, str],
    stdout_path: str,
    stderr_path: str,
    cwd: Optional[str] = None,
    preexec_fn=None,
) -> subprocess.Popen:
    os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
    stdout_f = open(stdout_path, "w", encoding="utf-8")
    stderr_f = open(stderr_path, "w", encoding="utf-8")
    return subprocess.Popen(
        cmd,
        env=env,
        stdout=stdout_f,
        stderr=stderr_f,
        cwd=cwd,
        preexec_fn=preexec_fn,
    )


def _kill_process_group(proc: subprocess.Popen, timeout_s: float = 10.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        pass
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.2)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass


def _request_terminate_process_group(
    proc: subprocess.Popen,
    timeout_s: float = 180.0,
    *,
    last_resort_force_kill: bool = True,
) -> None:
    """
    Gracefully terminate a process group with SIGTERM first.
    Only force-kill (SIGKILL) if the process group doesn't exit in timeout_s.
    """
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        return

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    if last_resort_force_kill:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _terminate_profiled_child_only(
    proc: subprocess.Popen,
    timeout_s: float = 180.0,
    *,
    last_resort_force_kill: bool = True,
) -> None:
    """
    Terminate only the profiled child (e.g. sglang server), NOT the profiler (nsys).
    Nsight Systems writes the .nsys-rep only after the profiled process exits; if we kill
    nsys (via killpg), the rep is never written or gets removed.
    """
    if proc.poll() is not None:
        return
    try:
        result = subprocess.run(
            ["pgrep", "-P", str(proc.pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            # No child found; fall back to process group (will kill nsys too).
            _request_terminate_process_group(
                proc, timeout_s=timeout_s, last_resort_force_kill=last_resort_force_kill
            )
            return
        child_pids = [int(x) for x in result.stdout.strip().splitlines() if x.strip()]
        for pid in child_pids:
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            except Exception:
                pass
    except Exception:
        _request_terminate_process_group(
            proc, timeout_s=timeout_s, last_resort_force_kill=last_resort_force_kill
        )
        return

    # Wait for nsys to finish export and exit (it exits after the child dies).
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            return
        time.sleep(0.2)

    # Timeout: force-kill children, then nsys as last resort.
    if last_resort_force_kill:
        try:
            result = subprocess.run(
                ["pgrep", "-P", str(proc.pid)],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                for pid_str in result.stdout.strip().splitlines():
                    try:
                        os.kill(int(pid_str), signal.SIGKILL)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass


def _pick_free_gpus(free_gpus: List[int], n: int) -> Optional[List[int]]:
    if len(free_gpus) < n:
        return None
    return sorted(free_gpus)[:n]


def _remove_gpus(free_gpus: List[int], gpus: Sequence[int]) -> None:
    free_set = set(gpus)
    free_gpus[:] = [g for g in free_gpus if g not in free_set]


def _add_gpus(free_gpus: List[int], gpus: Sequence[int]) -> None:
    free_gpus[:] = sorted(list(set(free_gpus).union(set(gpus))))


def _wait_for_http_ok(url: str, timeout_s: float = 300.0, interval_s: float = 1.0) -> None:
    # Avoid adding new deps; use urllib.
    import urllib.request
    import urllib.error

    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return
        except Exception as e:
            last_err = e
        time.sleep(interval_s)
    raise TimeoutError(f"Server not ready: {url}. last_err={last_err}")


def _http_post_best_effort(url: str, timeout_s: float = 15.0) -> bool:
    # Avoid adding new deps; use urllib.
    import urllib.request

    try:
        req = urllib.request.Request(url, data=b"{}", method="POST")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return 200 <= int(getattr(resp, "status", 0)) < 300
    except Exception:
        return False


def _parse_csv_float(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _aggregate_one_run(
    combined_csv_path: str,
    tp: int,
    input_len: int,
    gpu_clock: int,
    batch_size: int,
    output_lens: List[int],
) -> List[Dict[str, object]]:
    rows_out: List[Dict[str, object]] = []
    with open(combined_csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            op_name = (row.get("op_name") or "").strip()
            if not op_name:
                continue
            count = _parse_csv_float(row.get("count") or "")
            mean_us = _parse_csv_float(row.get("mean_us") or "")

            if op_name.startswith("P_"):
                # Replicate P latency for each output_len (prefill independent of decode length).
                for out_len in output_lens:
                    rows_out.append(
                        {
                            "tp": tp,
                            "input_len": input_len,
                            "output_len": out_len,
                            "gpu_clock": gpu_clock,
                            "batch_size": batch_size,
                            "stage": "P",
                            "op_name": op_name,
                            "count": count,
                            "latency_us": mean_us,
                        }
                    )
            elif op_name.startswith("D_"):
                for out_len in output_lens:
                    col = f"d_pos_{out_len}_us"
                    latency_us = _parse_csv_float(row.get(col) or "")
                    if latency_us is None:
                        continue
                    rows_out.append(
                        {
                            "tp": tp,
                            "input_len": input_len,
                            "output_len": out_len,
                            "gpu_clock": gpu_clock,
                            "batch_size": batch_size,
                            "stage": "D",
                            "op_name": op_name,
                            "count": count,
                            "latency_us": latency_us,
                        }
                    )
    return rows_out


def _merge_existing_results(
    *,
    work_dir: str,
    output_lens: List[int],
    final_csv_path: str,
    keep_processed: bool,
) -> str:
    """
    Merge existing per-run nvtx_PD_combined_stats.csv into a big table.
    Used by --only-merge to skip profiling.
    """
    out_rows: List[Dict[str, object]] = []

    processed_dirs = sorted(glob.glob(os.path.join(work_dir, "tp*", "processed")))
    if not processed_dirs:
        raise RuntimeError(f"No per-run processed dirs found under: {work_dir}")

    combined_paths: List[Tuple[str, int, int, int, int]] = []
    # (combined_csv_path, tp, input_len, gpu_clock, batch_size)

    for processed_dir in processed_dirs:
        combined_csv_path = os.path.join(processed_dir, "nvtx_PD_combined_stats.csv")
        if not os.path.exists(combined_csv_path):
            continue

        run_dir = os.path.dirname(processed_dir)
        base = os.path.basename(run_dir)
        # base like "tp1_in128_clk210_bs4"
        parts = base.split("_")
        if len(parts) < 3:
            raise RuntimeError(f"Cannot parse run metadata from dir: {base}")
        tp = int(parts[0].replace("tp", ""))
        input_len = int(parts[1].replace("in", ""))
        gpu_clock = int(parts[2].replace("clk", ""))
        batch_size = 1
        for part in parts[3:]:
            if part.startswith("bs"):
                batch_size = int(part.replace("bs", ""))
                break
        combined_paths.append((combined_csv_path, tp, input_len, gpu_clock, batch_size))

    if not combined_paths:
        raise RuntimeError(f"No per-run combined CSVs found under: {work_dir}")

    for combined_csv_path, tp, input_len, gpu_clock, batch_size in sorted(
        combined_paths, key=lambda x: (x[1], x[2], x[3], x[4])
    ):
        out_rows.extend(
            _aggregate_one_run(
                combined_csv_path,
                tp=tp,
                input_len=input_len,
                gpu_clock=gpu_clock,
                batch_size=batch_size,
                output_lens=output_lens,
            )
        )

    out_csv_path = os.path.abspath(final_csv_path)
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tp",
            "input_len",
            "output_len",
            "gpu_clock",
            "batch_size",
            "stage",
            "op_name",
            "count",
            "latency_us",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    if not keep_processed:
        for processed_dir in glob.glob(os.path.join(work_dir, "tp*", "processed")):
            try:
                shutil.rmtree(os.path.dirname(processed_dir), ignore_errors=True)
            except Exception:
                pass

    return out_csv_path


def _cleanup_nsys_artifacts(rep_prefix: str) -> None:
    # nsys outputs various sidecars; remove common ones.
    patterns = [
        rep_prefix + "*.nsys-rep",
        rep_prefix + "*.sqlite*",
        rep_prefix + "*.qdrep*",
        rep_prefix + "*.qdstrm*",
        rep_prefix + "*.ncu-rep*",
        rep_prefix + "*.nsys-cmd*",
    ]
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except Exception:
                # Best-effort cleanup.
                pass


def _cleanup_trace_csvs(out_dir: str) -> None:
    for p in glob.glob(os.path.join(out_dir, "nvtx_gpu_proj_trace*.csv")):
        try:
            os.remove(p)
        except Exception:
            pass


def _background_process_one_run(
    rep_file: str,
    processed_dir: str,
    d_bucket_count: int,
    d_pick_positions: str,
    d_num_cycles: int,
    *,
    nsys_python: str,
    out_combined_csv_name: str = "nvtx_PD_combined_stats.csv",
) -> None:
    os.makedirs(processed_dir, exist_ok=True)
    cmd = [
        nsys_python,
        "bash-test/nvtx_stats_from_rep.py",
        rep_file,
        "--d-bucket-count",
        str(d_bucket_count),
        "--d-pick-positions",
        d_pick_positions,
        "--d-num-cycles",
        str(d_num_cycles),
        "--out-dir",
        processed_dir,
    ]
    subprocess.run(cmd, cwd="/workspace/benchmark/sglang-main", check=True)

    # Cleanup nsys artifacts and intermediate trace csvs.
    rep_prefix = rep_file[: -len(".nsys-rep")] if rep_file.endswith(".nsys-rep") else rep_file
    _cleanup_nsys_artifacts(rep_prefix)
    _cleanup_trace_csvs(processed_dir)

    combined_csv = os.path.join(processed_dir, out_combined_csv_name)
    if not os.path.exists(combined_csv):
        raise FileNotFoundError(f"Combined CSV not found after processing: {combined_csv}")


def _run_pivot_wide(final_csv_path: str, bench_stage: str = "P") -> str:
    pivot_out_dir = os.path.join(os.path.dirname(os.path.abspath(final_csv_path)), "pivot_pd_ops_out")
    cmd = [
        sys.executable,
        "bash-test/pivot_pd_ops_wide.py",
        "--csv",
        os.path.abspath(final_csv_path),
        "--out-dir",
        pivot_out_dir,
        "--drop-p-output-len",
        "--stage",
        str(bench_stage).strip().upper(),
    ]
    subprocess.run(cmd, cwd="/workspace/benchmark/sglang-main", check=True)
    return pivot_out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch PD NVTX profiling for different TP/input_len/gpu_clock. Output_len is extracted via NVTX D-pick positions."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to pd_batch_config.json; values override CLI defaults.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="GPU id list, e.g. '0,1,2,3'. If empty, auto-detect from nvidia-smi.",
    )
    parser.add_argument(
        "--tp-list",
        type=str,
        default="1,2,4,8",
        help="TP values to test.",
    )
    parser.add_argument(
        "--input-lens",
        type=str,
        default="128,512,4096",
        help="Input lengths to test.",
    )
    parser.add_argument(
        "--output-lens",
        type=str,
        default="64,256,512",
        help=(
            "Decode lengths to extract from D. We will run bench with "
            "output_len=max(output-lens)+1 (prefill boundary produces one token)."
        ),
    )
    parser.add_argument(
        "--gpu-clocks",
        type=str,
        default="210,540,870,1200",
        help="GPU clocks to test (graphics/mem policy handled by bench_sglang.py).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/nvme1/models/llama3.1-8/",
    )
    parser.add_argument(
        "--port-base",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join(_CACHE_ROOT, "pd_batch_work"),
        help="Working directory for each run.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.path.join(_CACHE_ROOT, "pd_batch_logs"),
        help="Directory to store per-run merged logs.",
    )
    parser.add_argument(
        "--only-merge",
        action="store_true",
        help="Only merge existing processed CSVs into the big table (skip profiling).",
    )
    parser.add_argument(
        "--keep-processed",
        action="store_true",
        help="When using --only-merge, keep per-run processed dirs (do not delete).",
    )
    parser.add_argument(
        "--final-csv",
        type=str,
        default=os.path.join(_CACHE_ROOT, "pd_latency_big_table.csv"),
        help="Final big table CSV path.",
    )
    parser.add_argument(
        "--mem-clock",
        type=int,
        default=1593,
        help="mem_clock (MHz) passed to bench_sglang.py.",
    )
    parser.add_argument(
        "--num-seqs",
        type=int,
        default=None,
        help="Total number of requests per benchmark. If omitted, bench_sglang.py will default it to --batch_size.",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="1",
        help="Batch sizes to test in bench_sglang.py (comma-separated, e.g. '4,2,1').",
    )
    parser.add_argument(
        "--nsys-d-bucket-count",
        type=int,
        default=512,
        help="d-bucket-count for nvtx_stats_from_rep.py (usually equals max output_len).",
    )
    parser.add_argument(
        "--d-pick-positions",
        type=str,
        default="64,256,512",
        help="d-pick-positions for nvtx_stats_from_rep.py. Must be subset of [1..nsys-d-bucket-count].",
    )
    parser.add_argument(
        "--nsys-python",
        type=str,
        default=sys.executable,
        help="Python executable used to run nvtx_stats_from_rep.py.",
    )
    parser.add_argument(
        "--max-background-jobs",
        type=int,
        default=1,
        help=(
            "Max concurrent background nvtx_stats_from_rep jobs (each runs `nsys stats`). "
            "Default 1: parallel `nsys stats` often stalls or regresses progress on one session."
        ),
    )
    parser.add_argument(
        "--serialize-nsys-finalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Serialize nsys .nsys-rep export across parallel batch jobs (after bench, until rep ready). "
            "Two `nsys profile` sessions finalizing at once can hang or show erratic progress; default on."
        ),
    )
    parser.add_argument(
        "--server-start-timeout-s",
        type=int,
        default=420,
    )
    parser.add_argument(
        "--bench-timeout-s",
        type=int,
        default=3600 * 4,
    )
    parser.add_argument(
        "--nsys-rep-wait-timeout-s",
        type=int,
        default=10,
        help="bench结束后等待.nsys-rep写出(最多该时间)。",
    )
    parser.add_argument(
        "--nsys-rep-after-shutdown-wait-timeout-s",
        type=int,
        default=120,
        help="请求服务退出后继续等待.nsys-rep写出(最多该时间)。",
    )
    parser.add_argument(
        "--server-shutdown-timeout-s",
        type=int,
        default=180,
        help="请求服务退出后等待进程组退出(最多该时间)，超时才force-kill。",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["TTFT", "AF", "both", "ttft", "op"],
        default="AF",
        help=(
            "Benchmark mode. "
            "'TTFT': only stage metrics (TTFT/TPOT); "
            "'AF': only coarse per-op metrics (A/F); "
            "'both': run TTFT and AF as two separate variants, then merge."
        ),
    )
    parser.add_argument(
        "--bench-stage",
        type=str,
        default="D",
        choices=["P", "D", "p", "d"],
        help=(
            "P: prefill-only benchmark (max_new_tokens=0, no decode). "
            "D: decode proxy benchmark (max output_len + 1)."
        ),
    )
    parser.add_argument(
        "--sync-bench-num-iters",
        type=int,
        default=1,
        help="Internal A/F benchmark repeat count (env: SGLANG_SYNC_BENCH_NUM_ITERS).",
    )
    parser.add_argument(
        "--sync-bench-warmup-iters",
        type=int,
        default=0,
        help="Internal A/F benchmark warmup count (env: SGLANG_SYNC_BENCH_WARMUP_ITERS).",
    )
    parser.add_argument(
        "--bench-energy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable NVML energy (uJ per A/F op) in server-side qwen3 profiling "
            "(env: SGLANG_BENCH_ENERGY). JSON key: bench_energy."
        ),
    )
    args = parser.parse_args()
    cfg = _apply_json_config(args)
    mode_norm = str(args.mode).strip().lower()
    if mode_norm in {"ttft"}:
        args.mode = "TTFT"
    elif mode_norm in {"af", "op"}:
        args.mode = "AF"
    elif mode_norm == "both":
        args.mode = "both"
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")
    stage_norm = str(args.bench_stage).strip().upper()
    if stage_norm not in {"P", "D"}:
        raise ValueError(f"Unsupported bench stage: {args.bench_stage}")
    args.bench_stage = stage_norm
    if args.sync_bench_num_iters < 1:
        raise ValueError("--sync-bench-num-iters must be >= 1")
    if args.sync_bench_warmup_iters < 0:
        raise ValueError("--sync-bench-warmup-iters must be >= 0")

    # Parse lists.
    tp_list = [int(x) for x in args.tp_list.split(",") if x.strip()]
    input_lens = [int(x) for x in args.input_lens.split(",") if x.strip()]
    output_lens = [int(x) for x in args.output_lens.split(",") if x.strip()]
    gpu_clocks = [int(x) for x in args.gpu_clocks.split(",") if x.strip()]
    batch_sizes = [int(x) for x in args.batch_size.split(",") if x.strip()]
    # D stage uses one prefill boundary token (+1).
    # P stage runs prefill-only with no decode token.
    output_len_max = max(output_lens) + 1 if args.bench_stage == "D" else 0
    if args.num_seqs is not None and args.num_seqs <= 0:
        raise ValueError("--num-seqs must be >= 1 when provided")

    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.only_merge:
        out_csv_path = _merge_existing_results(
            work_dir=work_dir,
            output_lens=output_lens,
            final_csv_path=args.final_csv,
            keep_processed=bool(args.keep_processed),
        )
        print(f"[saved] big table: {out_csv_path}")
        print("[info] converting big table to wide pivot csv...")
        pivot_out_dir = _run_pivot_wide(out_csv_path, args.bench_stage)
        print(f"[saved] pivot dir: {pivot_out_dir}")
        return

    # Auto-detect GPUs if not provided.
    free_gpus: List[int] = []
    if args.gpus.strip():
        free_gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    else:
        # nvidia-smi -L returns lines like "GPU 0: ..."
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                cwd="/workspace/benchmark/sglang-main",
                capture_output=True,
                text=True,
                check=True,
            )
            ids = []
            for line in proc.stdout.splitlines():
                line = line.strip()
                if line.startswith("GPU "):
                    # Parse "GPU 0:"
                    mid = line.split(":")[0]
                    ids.append(int(mid.split()[1]))
            free_gpus = sorted(ids)
        except Exception as e:
            raise RuntimeError(
                "Failed to auto-detect GPUs. Please pass --gpus '0,1,2,...'."
            ) from e

    if not free_gpus:
        raise RuntimeError("No GPUs available.")

    # Generate job list. If --config includes runs/defaults, use explicit run list.
    jobs: List[JobSpec] = []
    if cfg and isinstance(cfg.get("runs"), list):
        defaults = cfg.get("defaults") or {}
        default_input = _to_int_list(defaults.get("input_len", input_lens))
        default_clock = _to_int_list(defaults.get("gpu_clock", gpu_clocks))
        default_bs = _to_int_list(defaults.get("batch_size", batch_sizes))
        for i, run in enumerate(cfg["runs"]):
            if not isinstance(run, dict) or "tp" not in run:
                raise ValueError(f"config runs[{i}] must be object and include tp")
            tp = int(run["tp"])
            run_inputs = _to_int_list(run.get("input_len", default_input))
            run_clocks = _to_int_list(run.get("gpu_clock", default_clock))
            run_bs = _to_int_list(run.get("batch_size", default_bs))
            for input_len in run_inputs:
                for gpu_clock in run_clocks:
                    for batch_size in run_bs:
                        jobs.append(
                            JobSpec(
                                tp=tp,
                                input_len=input_len,
                                gpu_clock=gpu_clock,
                                output_len_max=output_len_max,
                                batch_size=batch_size,
                            )
                        )
    else:
        # CLI Cartesian order: TP -> input_len -> GPU-clock.
        for tp in tp_list:
            for input_len in input_lens:
                for gpu_clock in gpu_clocks:
                    for batch_size in batch_sizes:
                        jobs.append(
                            JobSpec(
                                tp=tp,
                                input_len=input_len,
                                gpu_clock=gpu_clock,
                                output_len_max=output_len_max,
                                batch_size=batch_size,
                            )
                        )

    print(f"[info] discovered GPUs: {free_gpus}")
    print(f"[info] total jobs: {len(jobs)}")

    next_port = args.port_base
    port_step = 1

    # Background processing workers.
    background_executor = ThreadPoolExecutor(max_workers=args.max_background_jobs)
    background_futures = []
    # Nsight Systems can misbehave when multiple profile sessions finalize/export .nsys-rep concurrently.
    nsys_finalize_lock = threading.Lock()

    def job_worker(job_idx: int, job: JobSpec, assigned_gpus: List[int], port: int) -> None:
        tp, input_len, gpu_clock, batch_size = (
            job.tp,
            job.input_len,
            job.gpu_clock,
            job.batch_size,
        )
        run_id = f"tp{tp}_in{input_len}_clk{gpu_clock}_bs{batch_size}"
        run_dir = os.path.join(work_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        stdout_path = os.path.join(run_dir, "server_stdout.log")
        stderr_path = os.path.join(run_dir, "server_stderr.log")
        bench_stdout = os.path.join(run_dir, "bench_stdout.log")
        bench_stderr = os.path.join(run_dir, "bench_stderr.log")
        # Nsight Systems often writes sidecars (e.g. *.sqlite) using the output *basename* relative
        # to the process cwd. Using a fixed name like "sglang.out" across runs causes collisions when
        # cwd is shared (e.g. repo root). Use a unique basename + run nsys from run_dir.
        unique_tag = f"{run_id}_port{port}_{uuid.uuid4().hex[:8]}"
        profile_prefix = os.path.join(run_dir, unique_tag)
        rep_file = profile_prefix + ".nsys-rep"
        processed_dir = os.path.join(run_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        # Merge key outputs into a stable per-run log file so we can delete work_dir afterwards.
        merged_log_path = os.path.join(log_dir, f"{run_id}.log")
        merged_log_f = open(merged_log_path, "w", encoding="utf-8")
        merged_log_f.write(
            f"[pd-batch] run_id={run_id} assigned_gpus={assigned_gpus} port={port}\n"
        )
        merged_log_f.write(
            f"[pd-batch] input_len={input_len} output_len_max={job.output_len_max} tp={tp} gpu_clock={gpu_clock} batch_size={batch_size}\n"
        )
        merged_log_f.write(f"[pd-batch] model_path={args.model_path}\n")
        merged_log_f.write(f"[pd-batch] mode={args.mode}\n")
        merged_log_f.write(f"[pd-batch] nsys -o prefix (unique): {profile_prefix}\n")
        merged_log_f.flush()

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in assigned_gpus)
        variant_defs = []
        if args.mode == "TTFT":
            variant_defs = [("ttft", "1", "0")]
        elif args.mode == "AF":
            variant_defs = [("af", "0", "1")]
        elif args.mode == "both":
            # Two separate runs: stage-only then AF-only.
            variant_defs = [("ttft", "1", "0"), ("af", "0", "1")]
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
        # Ensure `python -m sglang.launch_server` resolves when nsys cwd is run_dir.
        py_path = os.path.join(_REPO_ROOT, "python")
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = f"{py_path}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = py_path

        # Start server directly (sync-op-bench dumps from llama.py).
        server_cmd = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            args.model_path,
            "--port",
            str(port),
            "--tensor-parallel-size",
            str(tp),
            "--disable-cuda-graph",
            "--disable-overlap-schedule",
            "--chunked-prefill-size",
            "-1",
            "--mem-fraction-static",
            "0.9",
        ]
        dump_glob = os.path.join(processed_dir, "*_rank*.json")
        proc = None
        print(
            f"[run] start server variants: {run_id} on GPUs={assigned_gpus}, port={port}"
        )

        try:
            # Shared-server mode: start once, run TTFT/AF sequentially.
            env_server = env.copy()
            if args.mode == "TTFT":
                env_server["SGLANG_SYNC_STAGE_BENCH"] = "1"
                env_server["SGLANG_SYNC_INTERNAL_OP_BENCH"] = "0"
            elif args.mode == "AF":
                env_server["SGLANG_SYNC_STAGE_BENCH"] = "0"
                env_server["SGLANG_SYNC_INTERNAL_OP_BENCH"] = "1"
            else:
                env_server["SGLANG_SYNC_STAGE_BENCH"] = "1"
                env_server["SGLANG_SYNC_INTERNAL_OP_BENCH"] = "1"
            env_server["SGLANG_SYNC_BENCH_REQUIRE_PROFILE_WINDOW"] = "0"
            env_server["SGLANG_SYNC_BENCH_NUM_ITERS"] = str(args.sync_bench_num_iters)
            env_server["SGLANG_SYNC_BENCH_WARMUP_ITERS"] = str(args.sync_bench_warmup_iters)
            env_server["SGLANG_BENCH_ENERGY"] = (
                "1" if bool(getattr(args, "bench_energy", True)) else "0"
            )
            env_server["SGLANG_TTFT_AF_CSV_ENABLE"] = "1"
            env_server["SGLANG_TTFT_AF_CSV_PATH"] = os.path.join(
                processed_dir, f"{unique_tag}_both_ttft_af.csv"
            )
            env_server["SGLANG_BENCH_TP"] = str(tp)
            env_server["SGLANG_BENCH_INPUT_LEN"] = str(input_len)
            env_server["SGLANG_BENCH_GPU_CLOCK"] = str(gpu_clock)
            env_server["SGLANG_BENCH_BATCH_SIZE"] = str(batch_size)
            env_server["SGLANG_OP_BENCH_DUMP_DIR"] = processed_dir
            phase_file = os.path.join(processed_dir, f"{unique_tag}_bench_phase.txt")
            env_server["SGLANG_BENCH_PHASE_FILE"] = phase_file

            # Boot with first variant phase.
            with open(phase_file, "w", encoding="utf-8") as f:
                f.write(variant_defs[0][0])

            proc = _run_cmd(
                server_cmd,
                env=env_server,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                cwd=run_dir,
                preexec_fn=os.setsid,
            )
            _wait_for_http_ok(
                f"http://127.0.0.1:{port}/model_info",
                timeout_s=args.server_start_timeout_s,
            )
            merged_log_f.write(f"[pd-batch] server ready at port={port}\n")
            merged_log_f.flush()

            for variant_idx, (variant_name, stage_bench, internal_op_bench) in enumerate(
                variant_defs
            ):
                with open(phase_file, "w", encoding="utf-8") as f:
                    f.write(variant_name)
                env_bench = env.copy()
                env_bench["CUDA_VISIBLE_DEVICES"] = env["CUDA_VISIBLE_DEVICES"]
                env_bench["SGLANG_BENCH_TP"] = str(tp)
                env_bench["SGLANG_BENCH_INPUT_LEN"] = str(input_len)
                env_bench["SGLANG_BENCH_GPU_CLOCK"] = str(gpu_clock)
                env_bench["SGLANG_BENCH_BATCH_SIZE"] = str(batch_size)
                env_bench["SGLANG_SYNC_STAGE_BENCH"] = stage_bench
                env_bench["SGLANG_SYNC_INTERNAL_OP_BENCH"] = internal_op_bench

                merged_log_f.write(
                    f"[pd-batch] start variant={variant_name} stage_bench={stage_bench} internal_op_bench={internal_op_bench}\n"
                )
                merged_log_f.flush()

                bench_cmd = [
                    sys.executable,
                    "bash-test/bench_sglang.py",
                    "--server-url",
                    f"http://127.0.0.1:{port}",
                ]
                if args.num_seqs is not None:
                    bench_cmd += ["--num_seqs", str(args.num_seqs)]
                bench_cmd += [
                    "--batch_size",
                    str(batch_size),
                    "--input_len",
                    str(input_len),
                    "--output_len",
                    str(job.output_len_max),
                    "--gpu_clock",
                    str(gpu_clock),
                    "--mem_clock",
                    str(args.mem_clock),
                    "--ignore_eos",
                ]
                with open(bench_stdout, "w", encoding="utf-8") as out_f, open(
                    bench_stderr, "w", encoding="utf-8"
                ) as err_f:
                    bench_proc = subprocess.Popen(
                        bench_cmd,
                        env=env_bench,
                        cwd="/workspace/benchmark/sglang-main",
                        stdout=out_f,
                        stderr=err_f,
                        preexec_fn=os.setsid,
                    )
                    try:
                        bench_proc.wait(timeout=args.bench_timeout_s)
                    except subprocess.TimeoutExpired:
                        _kill_process_group(bench_proc, timeout_s=10.0)
                        raise TimeoutError(f"bench timeout: {run_id}")

                merged_log_f.write(
                    f"[pd-batch] bench finished successfully variant={variant_name}\n"
                )
                merged_log_f.flush()

                # Between TTFT and AF in shared-server mode, clear KV cache.
                if variant_name == "ttft" and variant_idx + 1 < len(variant_defs):
                    ok = _http_post_best_effort(f"http://127.0.0.1:{port}/flush_cache")
                    merged_log_f.write(
                        f"[pd-batch] flush_cache after ttft: {'ok' if ok else 'failed'}\n"
                    )
                    merged_log_f.flush()

            merged_log_f.write("[pd-batch] stopping server, wait for json...\n")
            merged_log_f.flush()
            _request_terminate_process_group(
                proc,
                timeout_s=args.server_shutdown_timeout_s,
                last_resort_force_kill=True,
            )
            t0 = time.time()
            expected_dump_files = tp
            while time.time() - t0 < 60.0:
                if len(glob.glob(dump_glob)) >= expected_dump_files:
                    break
                time.sleep(0.5)
            dump_count = len(glob.glob(dump_glob))
            merged_log_f.write(
                f"[pd-batch] dump json ready: {dump_count} files (expected>={expected_dump_files})\n"
            )
            merged_log_f.flush()
            proc = None

        finally:
            # In case something went wrong, try to stop server process group.
            if proc is not None:
                _request_terminate_process_group(
                    proc,
                    timeout_s=10.0,
                    last_resort_force_kill=True,
                )
            merged_log_f.write("[pd-batch] job_worker exiting (cleanup)\n")
            merged_log_f.flush()
            try:
                merged_log_f.close()
            except Exception:
                pass

        # Append detailed stdout/stderr logs into merged log.
        # Note: run_dir may be deleted later; this ensures logs are preserved.
        for src_path, tag in [
            (stdout_path, "server_stdout"),
            (stderr_path, "server_stderr"),
            (bench_stdout, "bench_stdout"),
            (bench_stderr, "bench_stderr"),
        ]:
            if not os.path.exists(src_path):
                continue
            try:
                with open(merged_log_path, "ab") as out_b:
                    out_b.write(f"\n\n===== {tag}: {src_path} =====\n".encode("utf-8"))
                    with open(src_path, "rb") as in_b:
                        shutil.copyfileobj(in_b, out_b)
            except Exception:
                # Best-effort append; merged log header still exists.
                pass

        print(f"[run] sync-op dump finished: {run_id}")

    # Run in TP outer order; for each TP we partition GPUs into disjoint groups,
    # enabling parallel jobs within the same TP without dynamic GPU contention.
    for tp in tp_list:
        tp_jobs = [j for j in jobs if j.tp == tp]
        if not tp_jobs:
            continue

        if len(free_gpus) < tp:
            raise RuntimeError(f"Not enough GPUs for tp={tp}. need>={tp}, got={len(free_gpus)}")

        num_groups = len(free_gpus) // tp
        groups = [free_gpus[i * tp : (i + 1) * tp] for i in range(num_groups)]
        if not groups:
            raise RuntimeError(f"No GPU groups created for tp={tp}")

        group_queue: "Queue[List[int]]" = Queue()
        for g in groups:
            group_queue.put(g)

        print(f"[info] tp={tp}: groups={groups}")
        with ThreadPoolExecutor(max_workers=len(groups)) as tp_executor:
            tp_futures = []
            for tp_job in tp_jobs:
                port = next_port
                next_port += port_step

                # Preserve loop order inside this TP: input_len -> gpu_clock (as built).
                def _wrap(job_idx: int, job: JobSpec, port: int):
                    assigned_gpus = group_queue.get()
                    try:
                        job_worker(job_idx, job, assigned_gpus, port)
                    finally:
                        group_queue.put(assigned_gpus)

                # job_idx is the index in jobs list; used only for run logs.
                # We'll find it by reconstruction order.
                # This is safe as we only use it for uniqueness in log folder names.
                job_idx = jobs.index(tp_job)
                tp_futures.append(tp_executor.submit(_wrap, job_idx, tp_job, port))

            # Ensure all jobs for this TP complete (background processing may still be running).
            for fut in tp_futures:
                fut.result()

    # Wait for all background futures.
    print(f"[info] waiting background postprocess jobs: {len(background_futures)}")
    bg_errors = []
    for run_id, fut, _ in background_futures:
        try:
            fut.result()
        except Exception as e:
            bg_errors.append((run_id, str(e)))
    if bg_errors:
        raise RuntimeError(f"Background postprocess failed: {bg_errors[:5]} ... total={len(bg_errors)}")

    # Build big table from llama.py sync-op dumps.
    print("[info] building big table from sync-op dumps...")
    out_csv_path = os.path.abspath(args.final_csv)
    cmd = [
        sys.executable,
        "bash-test/sync_op_bench_dump_to_big_table.py",
        "--work-dir",
        work_dir,
        "--bench-stage",
        args.bench_stage,
        "--output-lens",
        args.output_lens,
        "--final-csv",
        out_csv_path,
    ]
    subprocess.run(cmd, cwd="/workspace/benchmark/sglang-main", check=True)
    print(f"[saved] big table: {out_csv_path}")
    print("[info] converting big table to wide pivot csv...")
    pivot_out_dir = _run_pivot_wide(out_csv_path, args.bench_stage)
    print(f"[saved] pivot dir: {pivot_out_dir}")

    # Cleanup run dirs except final outputs.
    # Keep intermediate artifacts when --keep-processed is enabled.
    if not args.keep_processed:
        for processed_dir in glob.glob(os.path.join(work_dir, "tp*", "processed")):
            try:
                shutil.rmtree(os.path.dirname(processed_dir), ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()

