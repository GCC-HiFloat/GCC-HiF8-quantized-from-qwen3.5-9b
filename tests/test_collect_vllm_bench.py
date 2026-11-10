"""CPU-only checks for the optional bench report collector."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "collector", Path(__file__).resolve().parents[1] / "scripts/collect_vllm_bench.py")
collector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(collector)


class MemoryParsingTests(unittest.TestCase):
    def test_selects_requested_board(self):
        raw = """| 5     910B3 | OK | 0 / 0 |
| 0 | 0000:40:00.0 | 0 | 0 / 0  1234 / 65536 |
| 6     910B3 | OK | 0 / 0 |
| 0 | 0000:41:00.0 | 0 | 0 / 0  3456 / 65536 |
"""
        self.assertEqual(collector.parse_hbm(raw, 6), {"used_mb": 3456, "capacity_mb": 65536})
        self.assertIsNone(collector.parse_hbm(raw, 7))

    def test_missing_or_failed_sample_is_not_zero(self):
        self.assertIsNone(collector.parse_hbm("npu-smi failed", 6))


if __name__ == "__main__":
    unittest.main()
