import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).parents[1] / "experiments/2026-10-06-cold-start/summarize.py"
spec = importlib.util.spec_from_file_location("summary", path)
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


class SummaryTests(unittest.TestCase):
    def row(self, mode, boot, fresh=True, latency=10):
        return {"mode": mode, "boot_id": boot, "fresh_worker": fresh,
                "status": "ok", "client_ttft_ms": latency}

    def test_snapshot_creation_unverified_and_warm_runs_are_excluded(self):
        rows = [self.row("gpu", "restored"), self.row("gpu", "creation"),
                self.row("gpu", "unknown"), self.row("gpu", "warm", False),
                {"mode": "gpu", "status": "error"}]
        evidence = {"restored": {"kind": "restore", "source": "log"},
                    "creation": {"kind": "creation", "source": "log"},
                    "warm": {"kind": "restore", "source": "log"}}
        result = summary.summarize(rows, evidence)["gpu"]
        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["excluded"], 4)
        self.assertEqual(result["errors"], 1)
        self.assertIsNone(result["p95_client_ttft_ms"])

    def test_duplicate_worker_cannot_inflate_sample_count(self):
        result = summary.summarize([self.row("none", "a"), self.row("none", "a"),
                                    self.row("none", "b", latency=30)], {})["none"]
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["median_client_ttft_ms"], 20)

    def test_p95_uses_100_eligible_samples(self):
        rows = [self.row("none", str(i), latency=i) for i in range(1, 101)]
        self.assertEqual(summary.summarize(rows, {})["none"]["p95_client_ttft_ms"], 95)


if __name__ == "__main__":
    unittest.main()
