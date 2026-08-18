import json

from sglang.srt.mem_cache.central_io_pressure import CentralIOPressureLogger


def test_pressure_logger_aggregates_events_without_one_record_per_operation(tmp_path):
    path = tmp_path / "pressure.jsonl"
    logger = CentralIOPressureLogger(path, model_id="hot", page_size=16, interval_s=1.0)

    state = {"live_pages": 3, "active_pages": 100, "clean_free_pages": 97, "draining_pages": 0}
    logger.record("alloc_pages", 3, **state, now_s=10.0)
    logger.record("backup_slots", 48, **state, now_s=10.4)
    logger.record("host_evict_slots", 16, **state, now_s=11.1)
    logger.record("restore_slots", 16, **state, now_s=11.1)
    logger.flush(**state, now_s=11.2)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["alloc_pages"] == 3
    assert records[0]["backup_slots"] == 48
    assert records[0]["restore_slots"] == 0
    assert records[1]["host_evict_slots"] == 16
    assert records[1]["restore_slots"] == 16
    assert records[1]["live_pages"] == 3
    assert records[1]["clean_free_pages"] == 97


def test_pressure_logger_is_noop_when_unconfigured():
    logger = CentralIOPressureLogger(None, model_id="hot", page_size=16)
    state = {"live_pages": 1, "active_pages": 10, "clean_free_pages": 9, "draining_pages": 0}
    logger.record("alloc_pages", 1, **state, now_s=1.0)
    logger.flush(**state, now_s=2.0)


def test_pressure_logger_records_retention_loss_feedback(tmp_path):
    path = tmp_path / "pressure.jsonl"
    logger = CentralIOPressureLogger(path, model_id="hot", page_size=16)
    state = {"live_pages": 1, "active_pages": 10, "clean_free_pages": 9, "draining_pages": 0}

    logger.record("retention_loss_tokens", 4096, **state, now_s=1.0)
    logger.flush(**state, now_s=1.1)

    record = json.loads(path.read_text())
    assert record["retention_loss_tokens"] == 4096
