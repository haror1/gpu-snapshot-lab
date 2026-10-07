import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

folder = Path(__file__).parents[1] / "experiments/2026-10-06-compiled-startup"
spec = importlib.util.spec_from_file_location("compiled_analysis", folder / "analyze.py")
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(folder))
spec.loader.exec_module(module)
sys.path.pop(0)


class AnalysisTests(unittest.TestCase):
    def test_captured_initialization_is_not_subtracted_from_restore_latency(self):
        rows = [{"variant": "eager-gpu", "mode": "gpu", "status": "ok",
                 "fresh_worker": True, "correct_token": True, "boot_id": boot,
                 "client_ttft_ms": latency, "server_ttft_ms": 20,
                 "capture_stages_ms": {"imports_ms": 900}, "post_restore_stages_ms": {}}
                for boot, latency in (("creation", 2000), ("restore-a", 500), ("restore-b", 600))]
        for row in rows:
            row["environment"] = {"gpu": "test GPU", "driver": "test driver"}
        evidence = {boot: {"kind": kind} for boot, kind in (
            ("creation", "creation"), ("restore-a", "restore"), ("restore-b", "restore"))}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            (directory / "evidence.json").write_text(json.dumps(evidence))
            result = module.analyze([directory])["eager-gpu"]
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["client_ttft_ms"]["median"], 550)
        self.assertEqual(result["outside_instrumented_stages_ms"]["median"], 530)
        self.assertEqual(result["initialization_stages_ms"]["imports_ms"]["n"], 1)
        self.assertEqual(result["initialization_provenance"], "snapshot_creation_runs")


if __name__ == "__main__":
    unittest.main()
