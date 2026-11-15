"""Small workload compatibility interface for the vendored allocator."""

MAX_RUN_S = 400


def wl_key(dataset, qps):
    return f"{dataset}_qps{qps}"


def workload_file(dataset, qps):
    raise RuntimeError("use explicit point.workload with the standalone CLI")
