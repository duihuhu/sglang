import json
import unittest
from unittest.mock import patch

from aflex_benchmark.collect.requests import stream_request


class _StreamingResponse:
    def __init__(self, chunks):
        self._lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks]
        self._lines.append(b"data: [DONE]\n")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def __iter__(self):
        return iter(self._lines)


def _request(chunks, output_len, times):
    req = {
        "request_id": "request-1",
        "input_len": 8,
        "output_len": output_len,
        "arrival_time_s": 0,
    }
    with patch(
        "aflex_benchmark.collect.requests.urllib.request.urlopen",
        return_value=_StreamingResponse(chunks),
    ), patch(
        "aflex_benchmark.collect.requests.time.monotonic", side_effect=times
    ):
        return stream_request("http://server", req, run_start=0)


class StreamRequestTests(unittest.TestCase):
    def test_cumulative_count_recovers_batched_af_tokens(self):
        result = _request(
            [
                {"text": "first", "meta_info": {"completion_tokens": 1}},
                {
                    "text": "cumulative backlog",
                    "meta_info": {
                        "completion_tokens": 256,
                        "ttft_pure_processing": 0.08,
                        "e2e_latency": 0.4,
                    },
                },
            ],
            256,
            [10.0, 10.0, 10.1, 10.5, 10.6],
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["completion_tokens"], 256)
        self.assertEqual(len(result["token_timestamps_s"]), 256)
        self.assertAlmostEqual(result["ttft_client_ms"], 100.0)
        self.assertGreater(result["tpot_ms"], 0)
        self.assertTrue(all(value >= 0 for value in result["itl_ms"]))
        self.assertAlmostEqual(result["token_timestamps_s"][-1], 10.4)

    def test_normal_cumulative_stream_keeps_observed_timestamps(self):
        result = _request(
            [
                {"text": "a", "meta_info": {"completion_tokens": 1}},
                {"text": "b", "meta_info": {"completion_tokens": 2}},
                {"text": "c", "meta_info": {"completion_tokens": 3}},
            ],
            3,
            [20.0, 20.0, 20.1, 20.3, 20.6, 20.7],
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["token_timestamps_s"], [20.1, 20.3, 20.6])
        self.assertAlmostEqual(result["itl_ms"][0], 200.0)
        self.assertAlmostEqual(result["itl_ms"][1], 300.0)

    def test_control_duplicates_and_retractions_do_not_overcount(self):
        result = _request(
            [
                {"text": "a", "meta_info": {"completion_tokens": 1}},
                {"text": "control", "meta_info": {"completion_tokens": 1}},
                {"text": "retract", "meta_info": {"completion_tokens": 0}},
                {"text": "control without count", "meta_info": {"retracted": True}},
                {"text": "ab", "meta_info": {"completion_tokens": 2}},
                {"text": "duplicate", "meta_info": {"completion_tokens": 2}},
            ],
            2,
            [30.0, 30.0, 30.1, 30.2, 30.3, 30.4, 30.5, 30.6, 30.7],
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["token_timestamps_s"], [30.1, 30.4])

    def test_single_batched_chunk_uses_server_ttft_and_e2e(self):
        result = _request(
            [{
                "text": "all tokens",
                "meta_info": {
                    "completion_tokens": 4,
                    "time_to_first_token_processing": 0.1,
                    "e2e_latency": 0.4,
                },
            }],
            4,
            [50.0, 50.0, 50.5, 50.6],
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["completion_tokens"], 4)
        self.assertAlmostEqual(result["ttft_client_ms"], 100.0)
        self.assertAlmostEqual(result["tpot_ms"], 100.0)
        self.assertEqual(result["token_timestamps_s"], [50.1, 50.2, 50.3, 50.4])

    def test_legacy_text_chunks_still_count_once_each(self):
        result = _request(
            [{"text": "a"}, {"text": "b"}],
            2,
            [40.0, 40.0, 40.1, 40.2, 40.3],
        )

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["completion_tokens"], 2)


if __name__ == "__main__":
    unittest.main()
