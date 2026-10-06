"""Regression: remote metadata must not require PyTorch on the caller."""
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

path = Path(__file__).parents[1] / "experiments/2026-10-06-cold-start/app.py"
spec = importlib.util.spec_from_file_location("benchmark_app", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class SpecialVersion(str):
    """Like torch.torch_version.TorchVersion, retains its library's type."""


class PayloadTests(unittest.TestCase):
    def test_resume_converts_library_version_to_builtin_string(self):
        worker = module.Worker()
        worker.capture_id = "capture"
        worker.torch = SimpleNamespace(
            __version__=SpecialVersion("2.6.0"),
            version=SimpleNamespace(cuda="12.4"),
            cuda=SimpleNamespace(get_device_name=lambda: "test GPU"),
        )
        with patch.object(Path, "read_text", return_value='{"revision":"test"}'), \
                patch("subprocess.check_output", side_effect=["driver\n", "torch==2.6.0\n"]), \
                patch("builtins.print"):
            worker.resume()
        self.assertIs(type(worker.environment["torch"]), str)
        self.assertEqual(worker.environment["torch"], "2.6.0")


if __name__ == "__main__":
    unittest.main()
