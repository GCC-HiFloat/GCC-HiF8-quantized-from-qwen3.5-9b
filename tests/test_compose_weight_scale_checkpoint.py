import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scripts.compose_weight_scale_checkpoint import compose
from vllm_ascend_hif8.codec import MAX_FINITE
from vllm_ascend_hif8.torch_codec import encode_tensor


class ComposeWeightScaleCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def _checkpoint(self, name, a_value, b_value, mode):
        root = self.root / name
        root.mkdir()
        shard = "model.safetensors"
        state = {
            "model.layers.0.self_attn.o_proj.weight": torch.full(
                (2, 3), a_value, dtype=torch.uint8
            ),
            "model.layers.0.self_attn.o_proj.weight_scale": torch.full(
                (2, 1), float(a_value), dtype=torch.float32
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.full(
                (2, 3), b_value, dtype=torch.uint8
            ),
            "model.layers.0.self_attn.q_proj.weight_scale": torch.full(
                (2, 1), float(b_value), dtype=torch.float32
            ),
        }
        save_file(state, root / shard)
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: shard for key in state}})
        )
        layers = {}
        for tensor_name in state:
            if tensor_name.endswith(".weight"):
                layers[tensor_name] = (
                    {"weight_scale_mode": mode} if mode is not None else {}
                )
        (root / "hif8_metadata.json").write_text(
            json.dumps(
                {
                    "format": "W8A8HIF8_STATIC",
                    "byte_layout": "ascend_hifloat8",
                    "quantized_layer_count": 2,
                    "weight_scale_scope": "self_attn",
                    "weight_scale_factor": 1.0,
                    "weight_scale_blend_alpha": 1.0,
                    "weight_activation_calibration": {"sha256": "abc"},
                    "layers": layers,
                }
            )
        )
        return root

    def test_copies_only_matching_weight_and_scale_payloads(self):
        base = self._checkpoint("base", 1, 2, None)
        donor = self._checkpoint("donor", 7, 8, "activation_mse")
        output = self.root / "output"
        count = compose(base, donor, output, r"\.o_proj\.weight$")
        self.assertEqual(count, 1)
        tensors = load_file(output / "model.safetensors")
        self.assertTrue(
            torch.equal(
                tensors["model.layers.0.self_attn.o_proj.weight"],
                torch.full((2, 3), 7, dtype=torch.uint8),
            )
        )
        self.assertTrue(
            torch.equal(
                tensors["model.layers.0.self_attn.q_proj.weight"],
                torch.full((2, 3), 2, dtype=torch.uint8),
            )
        )
        metadata = json.loads((output / "hif8_metadata.json").read_text())
        self.assertEqual(metadata["weight_scale_composition"]["selected_count"], 1)
        self.assertEqual(metadata["weight_scale_blend_alpha"], 1.0)
        self.assertEqual(metadata["weight_scale_blend_alphas"], [1.0])
        self.assertEqual(
            metadata["layers"]["model.layers.0.self_attn.o_proj.weight"][
                "weight_scale_mode"
            ],
            "activation_mse",
        )
        self.assertEqual(
            metadata["layers"]["model.layers.0.self_attn.q_proj.weight"][
                "weight_scale_mode"
            ],
            "amax",
        )

    def test_rejects_empty_selection_and_existing_destination(self):
        base = self._checkpoint("base", 1, 2, "amax")
        donor = self._checkpoint("donor", 7, 8, "activation_mse")
        with self.assertRaisesRegex(ValueError, "selected no"):
            compose(base, donor, self.root / "empty", r"\.k_proj\.weight$")
        existing = self.root / "existing"
        existing.mkdir()
        with self.assertRaises(FileExistsError):
            compose(base, donor, existing, r"\.o_proj\.weight$")

    def test_reencodes_only_matching_weight_with_geometric_scale(self):
        base = self._checkpoint("base", 1, 2, None)
        donor = self._checkpoint("donor", 4, 8, "activation_mse")
        source = self.root / "source"
        source.mkdir()
        shard = "model.safetensors"
        weights = {
            "model.layers.0.self_attn.o_proj.weight": torch.tensor(
                [[1.0, -0.5, 0.25], [0.75, -0.25, 0.5]], dtype=torch.bfloat16
            ),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(
                (2, 3), dtype=torch.bfloat16
            ),
        }
        save_file(weights, source / shard)
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: shard for key in weights}})
        )

        output = self.root / "blended"
        compose(
            base,
            donor,
            output,
            r"\.o_proj\.weight$",
            blend_alpha=0.5,
            float_source=source,
        )
        tensors = load_file(output / "model.safetensors")
        expected_scale = torch.full((2, 1), 2.0, dtype=torch.float32)
        expected_data = encode_tensor(
            (weights["model.layers.0.self_attn.o_proj.weight"].float() / 2.0).clamp(
                min=-MAX_FINITE, max=MAX_FINITE
            )
        )
        self.assertTrue(
            torch.equal(
                tensors["model.layers.0.self_attn.o_proj.weight_scale"],
                expected_scale,
            )
        )
        self.assertTrue(
            torch.equal(
                tensors["model.layers.0.self_attn.o_proj.weight"], expected_data
            )
        )
        self.assertTrue(
            torch.equal(
                tensors["model.layers.0.self_attn.q_proj.weight"],
                torch.full((2, 3), 2, dtype=torch.uint8),
            )
        )
        metadata = json.loads((output / "hif8_metadata.json").read_text())
        self.assertEqual(metadata["weight_scale_blend_alpha"], 0.5)
        self.assertEqual(metadata["weight_scale_blend_alphas"], [0.5])

    def test_payload_copy_preserves_donor_blend_provenance(self):
        base = self._checkpoint("base", 1, 2, None)
        donor = self._checkpoint("donor", 7, 8, "activation_mse")
        donor_metadata_path = donor / "hif8_metadata.json"
        donor_metadata = json.loads(donor_metadata_path.read_text())
        donor_metadata["weight_scale_blend_alpha"] = "mixed"
        donor_metadata["layers"][
            "model.layers.0.self_attn.o_proj.weight"
        ]["weight_scale_blend_alpha"] = 0.5
        donor_metadata["layers"][
            "model.layers.0.self_attn.q_proj.weight"
        ]["weight_scale_blend_alpha"] = 1.0
        donor_metadata_path.write_text(json.dumps(donor_metadata))

        output = self.root / "copied_blend"
        compose(base, donor, output, r"\.o_proj\.weight$")
        metadata = json.loads((output / "hif8_metadata.json").read_text())
        self.assertEqual(metadata["weight_scale_blend_alpha"], 0.5)
        self.assertEqual(metadata["weight_scale_blend_alphas"], [0.5])
        self.assertEqual(
            metadata["layers"]["model.layers.0.self_attn.o_proj.weight"][
                "weight_scale_blend_alpha"
            ],
            0.5,
        )


if __name__ == "__main__":
    unittest.main()
