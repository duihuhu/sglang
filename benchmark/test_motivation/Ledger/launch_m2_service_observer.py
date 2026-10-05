"""Benchmark-only server entrypoint. Observation is disabled unless requested."""

import os
import sys

if os.environ.get("M2_SERVICE_TRACE") == "1":
    from m2_service_observer import install_observer
    from sglang.srt.managers.schedule_policy import PrefillAdder, SchedulePolicy
    from sglang.srt.managers.scheduler import Scheduler

    install_observer(Scheduler, SchedulePolicy, PrefillAdder, enabled=True)

if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    try:
        run_server(prepare_server_args(sys.argv[1:]))
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
