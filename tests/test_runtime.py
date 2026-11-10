from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm_ascend_hif8 import runtime


class RuntimeOppPathTest(unittest.TestCase):
    def test_packaged_vendor_is_prepended_and_existing_paths_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary)
            vendor = package_root / "vendor" / "ascend910b" / "vendors" / "vllm_ascend_hif8"
            vendor.mkdir(parents=True)
            external = package_root / "external"

            with (
                patch.object(runtime, "_PACKAGE_ROOT", package_root),
                patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": str(external)}, clear=False),
            ):
                packaged = runtime.configure_opp_path()
                self.assertEqual(packaged, (vendor,))
                self.assertEqual(
                    os.environ["ASCEND_CUSTOM_OPP_PATH"].split(os.pathsep),
                    [str(vendor), str(external)],
                )

    def test_configuration_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary)
            vendor = package_root / "vendor" / "ascend910b" / "vendors" / "vllm_ascend_hif8"
            vendor.mkdir(parents=True)

            with (
                patch.object(runtime, "_PACKAGE_ROOT", package_root),
                patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": str(vendor)}, clear=False),
            ):
                runtime.configure_opp_path()
                runtime.configure_opp_path()
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], str(vendor))

    def test_other_packaged_targets_are_not_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            package_root = Path(temporary)
            foreign_vendor = package_root / "vendor" / "unsupported" / "vendors" / "vllm_ascend_hif8"
            foreign_vendor.mkdir(parents=True)
            with patch.object(runtime, "_PACKAGE_ROOT", package_root):
                self.assertEqual(runtime.packaged_vendor_roots(), ())

    def test_external_path_supports_source_tree_development(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(runtime, "_PACKAGE_ROOT", Path(temporary) / "package"),
                patch.dict(os.environ, {"ASCEND_CUSTOM_OPP_PATH": "/opt/custom/vendor"}, clear=False),
            ):
                self.assertEqual(runtime.configure_opp_path(), ())
                self.assertEqual(os.environ["ASCEND_CUSTOM_OPP_PATH"], "/opt/custom/vendor")

    def test_missing_packaged_and_external_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(runtime, "_PACKAGE_ROOT", Path(temporary) / "package"),
                patch.dict(os.environ, {}, clear=False),
            ):
                os.environ.pop("ASCEND_CUSTOM_OPP_PATH", None)
                with self.assertRaisesRegex(runtime.PackagedOppUnavailable, "payload is missing"):
                    runtime.configure_opp_path()


if __name__ == "__main__":
    unittest.main()
