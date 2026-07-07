#!/usr/bin/env python3
"""Real-time watchdog: detect stuck/crashed benchmarks and auto-restart.

Monitors:
  - run_pdaf_deploy_sweep_code.py (log stall > STALL_S)
  - run_best_pdaf_6scheme_benchmark.py
  - orchestrate_overnight.py (restart if dead)

Run on node34 alongside benchmarks.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [WATCHDOG] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("watchdog")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
LOG_DIR = MACRO_DIR.parent / "logs"
POLL_S = 60
STALL_S = 600
DEPLOY_STALL_S = 200  # deploy/pair line unchanged >200s -> stuck

NODE1 = os.environ.get("MN_NODE1_IP", "10.252.129.34")
NODE2 = os.environ.get("MN_NODE2_IP", "10.252.129.33")

SWEEP_LOG = LOG_DIR / "pdaf_deploy_sweep_code.log"
ORCH_LOG = LOG_DIR / "orchestrate_overnight.log"
BENCH_LOG = LOG_DIR / "best_pdaf_6scheme.log"
BENCH7_LOG = LOG_DIR / "7scheme_6dataset.log"
BENCH6_LOG = LOG_DIR / "6scheme_6dataset.log"
TIER1_MS_LOG = LOG_DIR / "tier1_megascale_aflex.log"
OTHER7_LOG = LOG_DIR / "7scheme_other_qps6.log"
WATCHDOG_LOG = LOG_DIR / "bench_watchdog.log"


def _pgrep(pattern: str) -> list[str]:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    if r.returncode != 0:
        return []
    return [x for x in r.stdout.strip().split("\n") if x]


def _log_idle(path: Path, stall_s: int) -> tuple[bool, float]:
    if not path.exists():
        return True, 0.0
    age = time.time() - path.stat().st_mtime
    return age > stall_s, age


def _last_log_line(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(errors="replace").splitlines()
        return lines[-1] if lines else ""
    except Exception:
        return ""


def _cleanup():
    log.warning("cleanup_all on both nodes")
    subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0,'%s'); import run_macro_benchmark as R; R.cleanup_all()"
         % MACRO_DIR],
        cwd=str(MACRO_DIR),
        timeout=120,
    )


def _kill_pattern(pattern: str):
    pids = _pgrep(pattern)
    for pid in pids:
        log.warning("kill pid=%s (%s)", pid, pattern)
        subprocess.run(["kill", "-9", pid], check=False)


def _start_sweep(resume: bool = True):
    args = [
        "nohup", "env",
        f"MN_NODE1_IP={NODE1}", f"MN_NODE2_IP={NODE2}",
        sys.executable, "-u",
        str(HERE / "run_pdaf_deploy_sweep_code.py"),
        "--skip-c3",
    ]
    if resume:
        args.append("--resume")
    args += [">", str(SWEEP_LOG), "2>&1", "<", "/dev/null", "&"]
    cmd = " ".join(args)
    log.info("restart sweep: %s", cmd)
    subprocess.Popen(cmd, shell=True, cwd=str(HERE))


def _start_orchestrate():
    cmd = (
        f"nohup env MN_NODE1_IP={NODE1} MN_NODE2_IP={NODE2} "
        f"{sys.executable} -u {HERE / 'orchestrate_overnight.py'} "
        f"> {ORCH_LOG} 2>&1 < /dev/null &"
    )
    log.info("restart orchestrate")
    subprocess.Popen(cmd, shell=True, cwd=str(HERE))


def _check_sweep_stuck() -> bool:
    if not _pgrep("run_pdaf_deploy_sweep_code.py"):
        return False
    idle, age = _log_idle(SWEEP_LOG, STALL_S)
    if not idle:
        return False
    last = _last_log_line(SWEEP_LOG)
    log.error("SWEEP STUCK: log idle %.0fs | last: %s", age, last[:120])
    return True


def _check_deploy_phase_stuck() -> bool:
    """Detect long wait on deploy without benchmark progress."""
    if not _pgrep("run_pdaf_deploy_sweep_code.py"):
        return False
    last = _last_log_line(SWEEP_LOG)
    if not last:
        return False
    # Health-wait hang: deploy warnings or no benchmark lines
    deploy_markers = ("Deploy C", "DEPLOY pdaf_", "pair ", "still launching")
    bench_markers = ("code_qps", "Thpt=", "Saved ")
    if any(m in last for m in bench_markers):
        return False
    if not any(m in last for m in deploy_markers):
        return False
    idle, age = _log_idle(SWEEP_LOG, DEPLOY_STALL_S)
    if idle:
        log.error("DEPLOY PHASE STUCK %.0fs: %s", age, last[:120])
        return True
    return False


def _recover_sweep():
    _kill_pattern("run_pdaf_deploy_sweep_code.py")
    _kill_pattern("launch_server")
    _kill_pattern("launch_router")
    time.sleep(5)
    _cleanup()
    time.sleep(3)
    _start_sweep(resume=True)


def _ensure_orchestrate():
    if _pgrep("orchestrate_overnight.py"):
        return
    if not _pgrep("run_pdaf_deploy_sweep_code.py") and not _pgrep(
            "run_best_pdaf_6scheme_benchmark.py"):
        return
    log.warning("orchestrate dead but work pending — restarting")
    _start_orchestrate()


def _check_bench_stuck() -> bool:
    for pattern, blog in (
        ("run_tier1_megascale_aflex_benchmark.py", TIER1_MS_LOG),
        ("run_6scheme_6dataset_benchmark.py", BENCH6_LOG),
        ("run_7scheme_6dataset_benchmark.py", BENCH7_LOG),
        ("run_7scheme_6dataset_benchmark.py", OTHER7_LOG),
        ("run_best_pdaf_6scheme_benchmark.py", BENCH_LOG),
    ):
        if not _pgrep(pattern):
            continue
        idle, age = _log_idle(blog, STALL_S)
        if idle:
            log.error("BENCH STUCK (%s): idle %.0fs", pattern, age)
            return True
    return False


def _recover_bench():
    if _pgrep("run_6scheme_6dataset_benchmark.py"):
        script = "run_6scheme_6dataset_benchmark.py"
        blog = BENCH6_LOG
    elif _pgrep("run_7scheme_6dataset_benchmark.py"):
        script = "run_7scheme_6dataset_benchmark.py"
        blog = BENCH7_LOG
    else:
        script = "run_best_pdaf_6scheme_benchmark.py"
        blog = BENCH_LOG
    _kill_pattern(script)
    _kill_pattern("launch_server")
    _kill_pattern("launch_router")
    time.sleep(5)
    _cleanup()
    time.sleep(3)
    cmd = (
        f"nohup env MN_NODE1_IP={NODE1} MN_NODE2_IP={NODE2} "
        f"{sys.executable} -u {HERE / script} "
        f"--resume > {blog} 2>&1 < /dev/null &"
    )
    subprocess.Popen(cmd, shell=True, cwd=str(HERE))


PAUSE_FLAG = HERE / "PAUSE_BENCH.flag"


def _paused() -> bool:
    return PAUSE_FLAG.exists()


def main():
    log.info("Watchdog started | poll=%ds stall=%ds deploy_stall=%ds",
             POLL_S, STALL_S, DEPLOY_STALL_S)
    sweep_restarts = 0
    while True:
        try:
            if _paused():
                log.info("PAUSE_BENCH.flag set — watchdog idle")
                time.sleep(POLL_S)
                continue
            if _check_sweep_stuck() or _check_deploy_phase_stuck():
                sweep_restarts += 1
                if sweep_restarts > 20:
                    log.error("too many sweep restarts, backing off 30min")
                    time.sleep(1800)
                    sweep_restarts = 0
                    continue
                _recover_sweep()
            elif _check_bench_stuck():
                _recover_bench()
            else:
                _ensure_orchestrate()
                if _pgrep("run_pdaf_deploy_sweep_code.py"):
                    idle, age = _log_idle(SWEEP_LOG, 99999)
                    log.info("sweep OK | log_age=%.0fs | %s",
                             age, _last_log_line(SWEEP_LOG)[:80])
                elif _pgrep("run_tier1_megascale_aflex_benchmark.py"):
                    idle, age = _log_idle(TIER1_MS_LOG, 99999)
                    log.info("tier1_ms_aflex OK | log_age=%.0fs", age)
                elif _pgrep("run_7scheme_6dataset_benchmark.py"):
                    blog = OTHER7_LOG if OTHER7_LOG.exists() else BENCH7_LOG
                    idle, age = _log_idle(blog, 99999)
                    log.info("7scheme OK | log_age=%.0fs | %s", age, blog.name)
                elif _pgrep("run_best_pdaf_6scheme_benchmark.py"):
                    idle, age = _log_idle(BENCH_LOG, 99999)
                    log.info("bench OK | log_age=%.0fs", age)
                elif _pgrep("orchestrate_overnight.py"):
                    log.info("orchestrate waiting")
                else:
                    finals = list((HERE / "results").glob(
                        "best_pdaf_6scheme_final_*.json"))
                    if finals:
                        log.info("all pipelines idle, final exists: %s",
                                 finals[-1].name)
                    else:
                        log.warning("nothing running — starting orchestrate")
                        _start_orchestrate()
        except Exception as e:
            log.exception("watchdog error: %s", e)
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
