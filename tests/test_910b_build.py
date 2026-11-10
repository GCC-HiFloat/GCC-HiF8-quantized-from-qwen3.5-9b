from __future__ import annotations

import ast
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class BuildTargetTest(unittest.TestCase):
    @staticmethod
    def selected_soc():
        tree = ast.parse((ROOT / "setup.py").read_text())
        names = {"_SUPPORTED_SOCS", "_DEFAULT_SOC"}
        nodes = [
            node for node in tree.body
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
            ) or (isinstance(node, ast.FunctionDef) and node.name == "selected_soc")
        ]
        namespace = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "setup.py", "exec"), namespace)
        return namespace["selected_soc"]()

    def test_default_and_explicit_target_are_910b(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HIF8_SOC", None)
            self.assertEqual(self.selected_soc(), "ascend910b")
            os.environ["HIF8_SOC"] = "ascend910b"
            self.assertEqual(self.selected_soc(), "ascend910b")

    def test_setup_rejects_other_targets(self):
        with patch.dict(os.environ, {"HIF8_SOC": "unsupported"}):
            with self.assertRaisesRegex(RuntimeError, "unsupported HIF8_SOC"):
                self.selected_soc()

    def test_build_script_rejects_other_targets_before_compilation(self):
        result = subprocess.run(
            ["bash", str(ROOT / "csrc/opp/build.sh"), "unsupported"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported SoC target", result.stderr)


if __name__ == "__main__":
    unittest.main()
