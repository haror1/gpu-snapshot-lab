import importlib.util
import unittest
from pathlib import Path

path = Path(__file__).parents[1] / "experiments/2026-10-06-compiled-startup/protocol.py"
spec = importlib.util.spec_from_file_location("compiled_protocol", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ProtocolTests(unittest.TestCase):
    def entry(self, container, message):
        return {"context_ids": [container], "message": message}

    def test_creation_then_restore_is_not_restore_only(self):
        entries = [self.entry("ta-one", "Creating GPU memory snapshot for Function."),
                   self.entry("ta-one", "Restoring Function from memory snapshot."),
                   self.entry("ta-one", '{"event":"worker_ready_hook","boot_id":"a","capture_id":"capture"}'),
                   self.entry("ta-two", "Restoring Function from memory snapshot."),
                   self.entry("ta-two", '{"event":"worker_ready_hook","boot_id":"b","capture_id":"capture"}')]
        evidence = module.classify_logs(entries)
        self.assertEqual(evidence["a"]["kind"], "creation")
        self.assertEqual(evidence["b"]["kind"], "restore")

    def test_restore_without_capture_lineage_stays_unverified(self):
        entries = [self.entry("ta-two", "Restoring Function from memory snapshot."),
                   self.entry("ta-two", '{"event":"worker_ready_hook","boot_id":"b","capture_id":"unknown"}')]
        self.assertEqual(module.classify_logs(entries)["b"]["kind"], "unverified")

    def test_created_message_is_creation_even_without_initial_creating_line(self):
        entries = [self.entry("ta-one", "Snapshot created. Restoring Function from memory snapshot."),
                   self.entry("ta-one", '{"event":"worker_ready_hook","boot_id":"a","capture_id":"capture"}')]
        self.assertEqual(module.classify_logs(entries)["a"]["kind"], "creation")

    def test_restore_evidence_cannot_validate_wrong_output_or_warm_reuse(self):
        row = {"mode": "gpu", "status": "ok", "fresh_worker": True,
               "correct_token": True, "boot_id": "b"}
        evidence = {"b": {"kind": "restore"}}
        self.assertTrue(module.eligible(row, evidence))
        self.assertFalse(module.eligible({**row, "correct_token": False}, evidence))
        self.assertFalse(module.eligible({**row, "fresh_worker": False}, evidence))
        self.assertFalse(module.eligible(row, {}))


if __name__ == "__main__":
    unittest.main()
