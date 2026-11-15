"""Self-contained, remotely executed ownership-checked session cleanup."""


def cleanup_session(
    session_id, owner, term_timeout=15.0, force=False, kill_timeout=5.0
):
    import os
    import signal
    import time
    from pathlib import Path

    token = ("AFLEX_BENCH_OWNER=" + owner).encode()

    def snapshot():
        members = {}
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
                state, group, session, start = (
                    fields[0],
                    int(fields[2]),
                    int(fields[3]),
                    fields[19],
                )
                if session != session_id or state == "Z":
                    continue
                owned = token in (entry / "environ").read_bytes().split(bytes([0]))
                members[int(entry.name)] = (group, start, owned)
            except (FileNotFoundError, ProcessLookupError):
                continue
        return members

    def signal_owned(sig):
        members = snapshot()
        if any(not item[2] for item in members.values()):
            raise RuntimeError("refusing to signal session with unverified ownership")
        for group in {item[0] for item in members.values()}:
            # Recheck the complete session immediately before each group signal.
            current = snapshot()
            if any(not item[2] for item in current.values()):
                raise RuntimeError("ownership changed before signal")
            if any(item[0] == group for item in current.values()):
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass

    def wait_exit(timeout):
        deadline = time.monotonic() + timeout
        while True:
            members = snapshot()
            if not members:
                return True
            if any(not item[2] for item in members.values()):
                raise RuntimeError("ownership mismatch while waiting")
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.1, max(0, deadline - time.monotonic())))

    if not snapshot():
        return
    signal_owned(signal.SIGTERM)
    if wait_exit(term_timeout):
        return
    if force:
        signal_owned(signal.SIGKILL)
        if wait_exit(kill_timeout):
            return
    raise RuntimeError("owned session still alive after cleanup timeout")
