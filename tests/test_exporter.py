import json
import tempfile
import unittest
from pathlib import Path

from vllm_ascend_hif8.exporter import (
    DYNAMIC_QUANT_TYPE,
    METADATA_FILENAME,
    QUANT_DESCRIPTION_FILENAME,
    QUANT_TYPE,
    STATIC_QUANT_TYPE,
    _canonical_activation_group,
    _weight_scale_scope_members,
    build_quant_description,
    expected_target_count,
    input_scale_name,
    is_qwen35_target_weight,
    quantize_checkpoint,
    scale_name,
)
from vllm_ascend_hif8.calibration import (
    CALIBRATION_FORMAT,
    CALIBRATION_SCHEMA_VERSION,
    file_sha256,
    qwen35_full_attention_layers,
    qwen35_linear_group_specs,
)


def qwen35_config(layer_types, hidden_size=8, intermediate_size=16):
    """Build a miniature Qwen3.5 config with the released nesting."""
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": len(layer_types),
            "layer_types": list(layer_types),
            "full_attention_interval": 4,
        },
    }


class ExporterSelectionTest(unittest.TestCase):
    def test_weight_scale_scope_accepts_module_combinations(self):
        self.assertEqual(
            _weight_scale_scope_members("self_attn+linear_attn"),
            {"self_attn", "linear_attn"},
        )
        self.assertEqual(
            _weight_scale_scope_members("all"),
            {"mlp", "self_attn", "linear_attn", "embedding"},
        )
        for invalid in ("", "mlp+", "mlp+mlp", "mlp+unknown", "all+mlp"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _weight_scale_scope_members(invalid)

    def test_selects_released_qwen35_linear_scope(self):
        targets = [
            "model.language_model.layers.3.self_attn.q_proj.weight",
            "model.language_model.layers.3.self_attn.k_proj.weight",
            "model.language_model.layers.3.self_attn.v_proj.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
            "model.language_model.layers.0.mlp.gate_proj.weight",
            "model.language_model.layers.0.mlp.up_proj.weight",
            "model.language_model.layers.0.mlp.down_proj.weight",
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            "model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "model.language_model.layers.0.linear_attn.out_proj.weight",
            "model.language_model.embed_tokens.weight",
        ]
        self.assertTrue(all(is_qwen35_target_weight(name) for name in targets))

    def test_selects_text_only_layout_without_language_model_level(self):
        self.assertTrue(is_qwen35_target_weight("model.layers.0.mlp.down_proj.weight"))
        self.assertTrue(is_qwen35_target_weight("model.layers.7.self_attn.q_proj.weight"))

    def test_excludes_sensitive_and_non_linear_weights(self):
        excluded = [
            "model.language_model.norm.weight",
            "lm_head.weight",
            "model.language_model.layers.0.input_layernorm.weight",
            "model.language_model.layers.0.self_attn.q_norm.weight",
            "model.language_model.layers.0.linear_attn.in_proj_b.weight",
            "model.language_model.layers.0.linear_attn.in_proj_a.weight",
            "model.language_model.layers.0.linear_attn.conv1d.weight",
            "model.language_model.layers.0.linear_attn.norm.weight",
            "model.visual.blocks.0.mlp.linear_fc1.weight",
            "model.visual.blocks.0.attn.qkv.weight",
            "model.visual.merger.linear_fc1.weight",
        ]
        self.assertFalse(any(is_qwen35_target_weight(name) for name in excluded))

    def test_excludes_the_speculative_mtp_block(self):
        excluded = [
            "mtp.layers.0.self_attn.q_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight",
            "mtp.fc.weight",
        ]
        self.assertFalse(any(is_qwen35_target_weight(name) for name in excluded))

    def test_rejects_target_leaf_under_a_foreign_module(self):
        self.assertFalse(is_qwen35_target_weight("model.layers.0.mlp.q_proj.weight"))
        self.assertFalse(is_qwen35_target_weight("model.layers.0.self_attn.out_proj.weight"))
        self.assertFalse(is_qwen35_target_weight("model.layers.0.mlp.down_proj.bias"))

    def test_quant_description_marks_unquantized_weights_float(self):
        target = "model.language_model.layers.0.mlp.down_proj.weight"
        norm = "model.language_model.layers.0.input_layernorm.weight"
        description = build_quant_description((target, norm, "model.language_model.layers.0.mlp.down_proj.bias"))
        self.assertEqual(description, {target: QUANT_TYPE, norm: "FLOAT"})

    def test_scale_name_preserves_layer_prefix(self):
        self.assertEqual(
            scale_name("model.language_model.layers.0.mlp.down_proj.weight"),
            "model.language_model.layers.0.mlp.down_proj.weight_scale",
        )
        with self.assertRaises(ValueError):
            scale_name("model.language_model.layers.0.mlp.down_proj.bias")

    def test_maps_fused_members_to_one_activation_group(self):
        prefix = "model.language_model.layers.3"
        self.assertEqual(
            _canonical_activation_group(f"{prefix}.self_attn.k_proj.weight"),
            f"{prefix}.self_attn.q_proj",
        )
        self.assertEqual(
            _canonical_activation_group(f"{prefix}.mlp.up_proj.weight"),
            f"{prefix}.mlp.gate_proj",
        )
        linear_prefix = "model.language_model.layers.0"
        self.assertEqual(
            _canonical_activation_group(
                f"{linear_prefix}.linear_attn.in_proj_z.weight"
            ),
            f"{linear_prefix}.linear_attn.in_proj_qkv",
        )
        self.assertIsNone(
            _canonical_activation_group(
                "model.language_model.embed_tokens.weight"
            )
        )


class ExpectedTargetCountTest(unittest.TestCase):
    def test_matches_released_qwen35_9b_layer_pattern(self):
        layer_types = ["linear_attention"] * 32
        for index in range(3, 32, 4):
            layer_types[index] = "full_attention"
        self.assertEqual(layer_types.count("full_attention"), 8)
        # 8 x 4 attention + 24 x 3 GDN + 32 x 3 MLP + 1 embedding
        self.assertEqual(expected_target_count(qwen35_config(layer_types)), 201)

    def test_derives_layer_types_from_full_attention_interval(self):
        config = qwen35_config(["full_attention"] * 8)
        del config["text_config"]["layer_types"]
        # 2 full-attention + 6 GDN + 8 MLP
        self.assertEqual(expected_target_count(config), 2 * 4 + 6 * 3 + 8 * 3 + 1)

    def test_supports_a_flat_text_only_config(self):
        config = {"num_hidden_layers": 2, "layer_types": ["full_attention", "linear_attention"]}
        self.assertEqual(expected_target_count(config), 4 + 3 + 2 * 3 + 1)

    def test_rejects_unusable_configs(self):
        with self.assertRaises(ValueError):
            expected_target_count({"num_hidden_layers": 0})
        with self.assertRaises(ValueError):
            expected_target_count(qwen35_config(["full_attention", "sliding_attention"]))
        short = qwen35_config(["full_attention", "linear_attention"])
        short["text_config"]["num_hidden_layers"] = 3
        with self.assertRaises(ValueError):
            expected_target_count(short)


class CheckpointExportTest(unittest.TestCase):
    """End-to-end export over a miniature Qwen3.5-shaped checkpoint."""

    LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention")
    HIDDEN = 8
    INTERMEDIATE = 16

    def setUp(self):
        self.torch = __import__("torch")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def build_state(self):
        torch = self.torch
        hidden, inter = self.HIDDEN, self.INTERMEDIATE
        generator = torch.Generator().manual_seed(0)

        def randn(*shape):
            return torch.randn(*shape, generator=generator, dtype=torch.float32).to(torch.bfloat16)

        state = {
            "model.language_model.embed_tokens.weight": randn(32, hidden),
            "model.language_model.norm.weight": randn(hidden),
            "lm_head.weight": randn(32, hidden),
        }
        for index, kind in enumerate(self.LAYER_TYPES):
            prefix = f"model.language_model.layers.{index}"
            state[f"{prefix}.input_layernorm.weight"] = randn(hidden)
            state[f"{prefix}.post_attention_layernorm.weight"] = randn(hidden)
            state[f"{prefix}.mlp.gate_proj.weight"] = randn(inter, hidden)
            state[f"{prefix}.mlp.up_proj.weight"] = randn(inter, hidden)
            state[f"{prefix}.mlp.down_proj.weight"] = randn(hidden, inter)
            if kind == "full_attention":
                state[f"{prefix}.self_attn.q_proj.weight"] = randn(hidden, hidden)
                state[f"{prefix}.self_attn.k_proj.weight"] = randn(hidden, hidden)
                state[f"{prefix}.self_attn.v_proj.weight"] = randn(hidden, hidden)
                state[f"{prefix}.self_attn.o_proj.weight"] = randn(hidden, hidden)
                state[f"{prefix}.self_attn.q_norm.weight"] = randn(hidden)
            else:
                state[f"{prefix}.linear_attn.in_proj_qkv.weight"] = randn(2 * hidden, hidden)
                state[f"{prefix}.linear_attn.in_proj_z.weight"] = randn(hidden, hidden)
                state[f"{prefix}.linear_attn.out_proj.weight"] = randn(hidden, hidden)
                state[f"{prefix}.linear_attn.in_proj_a.weight"] = randn(hidden, hidden)
                state[f"{prefix}.linear_attn.in_proj_b.weight"] = randn(hidden, hidden)
                state[f"{prefix}.linear_attn.conv1d.weight"] = randn(hidden, 1, 4)
        return state

    def write_checkpoint(self, *, sharded):
        from safetensors.torch import save_file

        source = self.root / ("sharded" if sharded else "single")
        source.mkdir(parents=True)
        state = self.build_state()
        config = qwen35_config(self.LAYER_TYPES, self.HIDDEN, self.INTERMEDIATE)
        (source / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        (source / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

        if not sharded:
            save_file(state, source / "model.safetensors")
            return source, state

        names = sorted(state)
        half = len(names) // 2
        groups = {
            "model-00001-of-00002.safetensors": names[:half],
            "model-00002-of-00002.safetensors": names[half:],
        }
        weight_map = {}
        for shard, members in groups.items():
            save_file({name: state[name] for name in members}, source / shard)
            weight_map.update({name: shard for name in members})
        (source / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 1}, "weight_map": weight_map}, indent=2),
            encoding="utf-8",
        )
        return source, state

    def calibration_artifact(self, source, state):
        config = json.loads((source / "config.json").read_text())
        config["text_config"]["num_key_value_heads"] = 2
        # Keep the source and its identity hash in sync with the config used by
        # this static-export test.
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        specs = qwen35_linear_group_specs(self.LAYER_TYPES)
        index = source / "model.safetensors.index.json"
        identity_names = ["config.json"]
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            identity_names += [
                "model.safetensors.index.json",
                *sorted(set(weight_map.values())),
            ]
        else:
            identity_names.append("model.safetensors")
        groups = {
            spec.canonical_checkpoint_prefix: {
                "scale": 0.01 + ordinal * 0.0001,
                "checkpoint_input_scale": input_scale_name(
                    spec.canonical_checkpoint_prefix
                ),
                "observer": {
                    "channel_mean_square": [
                        1.0
                    ]
                    * state[
                        f"{spec.canonical_checkpoint_prefix}.weight"
                    ].shape[1]
                },
            }
            for ordinal, spec in enumerate(specs)
        }
        kv_layers = {
            str(index): {
                "key": {"scales": [0.1, 0.2]},
                "value": {"scales": [0.3, 0.4]},
            }
            for index in qwen35_full_attention_layers(self.LAYER_TYPES)
        }
        return {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "format": CALIBRATION_FORMAT,
            "model": {
                "model_type": config["model_type"],
                "identity_sha256": {
                    name: file_sha256(source / name) for name in identity_names
                },
            },
            "linear_groups": groups,
            "kv_cache": {"layers": kv_layers},
        }

    def assert_quantized_checkpoint(self, source, output, state):
        from safetensors.torch import load_file

        torch = self.torch
        expected_targets = {name for name in state if is_qwen35_target_weight(name)}
        self.assertEqual(len(expected_targets), expected_target_count(qwen35_config(self.LAYER_TYPES)))

        exported = {}
        for shard in sorted(output.glob("*.safetensors")):
            exported.update(load_file(shard, device="cpu"))

        self.assertEqual(set(exported), set(state) | {scale_name(name) for name in expected_targets})
        for name in expected_targets:
            weight, scale = exported[name], exported[scale_name(name)]
            self.assertEqual(weight.dtype, torch.uint8, name)
            self.assertEqual(tuple(weight.shape), tuple(state[name].shape), name)
            self.assertEqual(scale.dtype, torch.float32, name)
            self.assertEqual(tuple(scale.shape), (state[name].shape[0], 1), name)
            self.assertTrue(bool(torch.all(scale > 0)), name)
        for name in set(state) - expected_targets:
            self.assertEqual(exported[name].dtype, torch.bfloat16, name)
            self.assertTrue(bool(torch.equal(exported[name], state[name])), name)

        description = json.loads((output / QUANT_DESCRIPTION_FILENAME).read_text())
        self.assertEqual({name for name, kind in description.items() if kind == QUANT_TYPE}, expected_targets)
        self.assertTrue(all(kind == "FLOAT" for name, kind in description.items() if name not in expected_targets))

        metadata = json.loads((output / METADATA_FILENAME).read_text())
        self.assertEqual(metadata["format"], QUANT_TYPE)
        self.assertEqual(metadata["byte_layout"], "ascend_hifloat8")
        self.assertEqual(metadata["quantized_layer_count"], len(expected_targets))
        self.assertEqual(set(metadata["layers"]), expected_targets)

        config = json.loads((output / "config.json").read_text())
        # vllm-ascend detects the checkpoint through quant_model_description.json
        # and only loads it while quant_description is empty, so config.json must
        # stay free of a quantization_config section.
        self.assertNotIn("quantization_config", config)
        self.assertEqual(config["text_config"]["num_hidden_layers"], len(self.LAYER_TYPES))
        self.assertTrue((output / "tokenizer_config.json").exists())

    def test_exports_a_single_file_checkpoint(self):
        source, state = self.write_checkpoint(sharded=False)
        output = self.root / "single-out"
        metadata = quantize_checkpoint(source, output)
        self.assertEqual(metadata["quantized_layer_count"], expected_target_count(qwen35_config(self.LAYER_TYPES)))
        self.assertFalse((output / "model.safetensors.index.json").exists())
        self.assert_quantized_checkpoint(source, output, state)

    def test_exports_a_sharded_checkpoint_and_rewrites_the_index(self):
        source, state = self.write_checkpoint(sharded=True)
        output = self.root / "sharded-out"
        quantize_checkpoint(source, output)
        self.assert_quantized_checkpoint(source, output, state)

        index = json.loads((output / "model.safetensors.index.json").read_text())
        weight_map = index["weight_map"]
        source_map = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
        for name in state:
            self.assertEqual(weight_map[name], source_map[name], name)
            if is_qwen35_target_weight(name):
                self.assertEqual(weight_map[scale_name(name)], source_map[name], name)
        self.assertEqual(len(weight_map), len(source_map) + metadata_target_count(output))
        self.assertGreater(index["metadata"]["total_size"], 0)

    def test_exports_static_linear_and_kv_scales_from_calibration(self):
        from safetensors.torch import load_file

        source, state = self.write_checkpoint(sharded=True)
        calibration = self.calibration_artifact(source, state)
        output = self.root / "static-out"
        metadata = quantize_checkpoint(
            source, output, calibration=calibration
        )

        self.assertEqual(metadata["format"], STATIC_QUANT_TYPE)
        self.assertEqual(
            metadata["activation_quantization"],
            "static_per_linear_input_tensor",
        )
        self.assertEqual(metadata["static_activation_scale_count"], 16)
        self.assertEqual(metadata["calibration"], calibration)
        self.assertEqual(
            metadata["kv_cache"]["layers"]["3"]["key_scales"],
            [0.1, 0.2],
        )

        description = json.loads(
            (output / QUANT_DESCRIPTION_FILENAME).read_text()
        )
        embedding = "model.language_model.embed_tokens.weight"
        self.assertEqual(description[embedding], DYNAMIC_QUANT_TYPE)
        for name in state:
            if is_qwen35_target_weight(name) and name != embedding:
                self.assertEqual(description[name], STATIC_QUANT_TYPE, name)

        index = json.loads(
            (output / "model.safetensors.index.json").read_text()
        )
        exported = {}
        for shard in sorted(output.glob("*.safetensors")):
            exported.update(load_file(shard, device="cpu"))
        expected_scale_names = {
            entry["checkpoint_input_scale"]
            for entry in calibration["linear_groups"].values()
        }
        self.assertEqual(
            expected_scale_names,
            {name for name in exported if name.endswith(".input_scale")},
        )
        self.assertTrue(expected_scale_names <= set(index["weight_map"]))
        for prefix, entry in calibration["linear_groups"].items():
            tensor = exported[input_scale_name(prefix)]
            self.assertEqual(tensor.dtype, self.torch.float32)
            self.assertEqual(tuple(tensor.shape), (1,))
            self.assertAlmostEqual(float(tensor.item()), entry["scale"], places=7)

    def test_exports_activation_aware_fixed_weight_scales(self):
        source, state = self.write_checkpoint(sharded=True)
        calibration = self.calibration_artifact(source, state)
        output = self.root / "activation-mse-out"
        metadata = quantize_checkpoint(
            source,
            output,
            calibration=calibration,
            weight_scale_mode="activation_mse",
            weight_scale_scope="self_attn",
            weight_activation_calibration=calibration,
        )
        self.assertEqual(metadata["weight_scale_mode"], "activation_mse")
        self.assertEqual(
            metadata["weight_activation_calibration"]["group_count"], 16
        )
        for name, entry in metadata["layers"].items():
            expected = "activation_mse" if ".self_attn." in name else "amax"
            self.assertEqual(entry["weight_scale_mode"], expected, name)

    def test_activation_aware_scale_supports_fixed_blend_and_name_filter(self):
        source, state = self.write_checkpoint(sharded=True)
        calibration = self.calibration_artifact(source, state)
        output = self.root / "activation-mse-filtered-out"
        metadata = quantize_checkpoint(
            source,
            output,
            calibration=calibration,
            weight_scale_mode="activation_mse",
            weight_scale_scope="self_attn",
            weight_activation_calibration=calibration,
            weight_scale_blend_alpha=0.25,
            weight_scale_name_regex=r"\.q_proj\.weight$",
        )
        self.assertEqual(metadata["weight_scale_blend_alpha"], 0.25)
        self.assertEqual(metadata["weight_scale_name_regex"], r"\.q_proj\.weight$")
        for name, entry in metadata["layers"].items():
            expected = (
                "activation_mse"
                if ".self_attn." in name and name.endswith(".q_proj.weight")
                else "amax"
            )
            self.assertEqual(entry["weight_scale_mode"], expected, name)

    def test_static_export_rejects_wrong_or_incomplete_provenance(self):
        source, state = self.write_checkpoint(sharded=True)
        calibration = self.calibration_artifact(source, state)

        broken = json.loads(json.dumps(calibration))
        broken["model"]["identity_sha256"].pop(
            "model-00001-of-00002.safetensors"
        )
        with self.assertRaisesRegex(ValueError, "missing hashes"):
            quantize_checkpoint(
                source, self.root / "missing-hash-out", calibration=broken
            )

        broken = json.loads(json.dumps(calibration))
        broken["model"]["identity_sha256"]["config.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            quantize_checkpoint(
                source, self.root / "wrong-hash-out", calibration=broken
            )

        broken = json.loads(json.dumps(calibration))
        broken["linear_groups"].pop(next(iter(broken["linear_groups"])))
        with self.assertRaisesRegex(ValueError, "do not exactly cover"):
            quantize_checkpoint(
                source, self.root / "missing-group-out", calibration=broken
            )

    def test_rejects_a_partial_checkpoint(self):
        source, _ = self.write_checkpoint(sharded=False)
        config = json.loads((source / "config.json").read_text())
        config["text_config"]["num_hidden_layers"] += 1
        config["text_config"]["layer_types"].append("full_attention")
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing to emit a partial checkpoint"):
            quantize_checkpoint(source, self.root / "partial-out")

    def test_rejects_a_non_empty_output_directory(self):
        source, _ = self.write_checkpoint(sharded=False)
        output = self.root / "occupied"
        output.mkdir()
        (output / "stale.bin").write_bytes(b"")
        with self.assertRaises(FileExistsError):
            quantize_checkpoint(source, output)

    def test_rejects_a_checkpoint_without_a_config(self):
        source, _ = self.write_checkpoint(sharded=False)
        (source / "config.json").unlink()
        with self.assertRaises(FileNotFoundError):
            quantize_checkpoint(source, self.root / "no-config-out")

    def test_rejects_an_already_quantized_input(self):
        source, _ = self.write_checkpoint(sharded=False)
        config = json.loads((source / "config.json").read_text())
        config["quantization_config"] = {"quant_method": "compressed-tensors"}
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already has quantization_config"):
            quantize_checkpoint(source, self.root / "requant-out")


class WeightScaleSearchTest(unittest.TestCase):
    def test_mse_search_is_fixed_and_not_worse_than_amax_on_its_grid(self):
        import torch

        from vllm_ascend_hif8.codec import ASCEND_DECODE_TABLE
        from vllm_ascend_hif8.torch_codec import (
            quantize_tensor_rows,
            quantize_tensor_rows_activation_mse,
            quantize_tensor_rows_mse,
        )

        values = torch.tensor(
            [
                [8.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0],
                [0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0],
            ],
            dtype=torch.bfloat16,
        )
        amax_data, amax_scale = quantize_tensor_rows(values)
        mse_data, mse_scale = quantize_tensor_rows_mse(values)
        lut = torch.tensor(ASCEND_DECODE_TABLE, dtype=torch.float32)
        source = values.to(torch.float32)
        amax_error = (
            lut[amax_data.long()] * amax_scale[:, None] - source
        ).square().mean(dim=1)
        mse_error = (
            lut[mse_data.long()] * mse_scale[:, None] - source
        ).square().mean(dim=1)

        self.assertEqual(mse_data.dtype, torch.uint8)
        self.assertEqual(mse_scale.dtype, torch.float32)
        self.assertEqual(tuple(mse_scale.shape), (2,))
        self.assertTrue(bool(torch.isfinite(mse_scale).all()))
        self.assertTrue(bool((mse_scale > 0).all()))
        self.assertTrue(bool((mse_error <= amax_error + 1.0e-12).all()))

        importance = torch.tensor(
            [1000.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        weighted_data, weighted_scale = quantize_tensor_rows_activation_mse(
            values, importance
        )
        weighted_error = (
            (
                lut[weighted_data.long()] * weighted_scale[:, None]
                - source
            ).square()
            * importance
        ).sum(dim=1) / importance.sum()
        amax_weighted_error = (
            (lut[amax_data.long()] * amax_scale[:, None] - source).square()
            * importance
        ).sum(dim=1) / importance.sum()
        self.assertEqual(weighted_data.dtype, torch.uint8)
        self.assertEqual(weighted_scale.dtype, torch.float32)
        self.assertTrue(
            bool((weighted_error <= amax_weighted_error + 1.0e-12).all())
        )

        amax_blend_data, amax_blend_scale = quantize_tensor_rows_activation_mse(
            values, importance, blend_alpha=0.0
        )
        self.assertTrue(bool(torch.equal(amax_blend_data, amax_data)))
        self.assertTrue(bool(torch.equal(amax_blend_scale, amax_scale)))

        half_data, half_scale = quantize_tensor_rows_activation_mse(
            values, importance, blend_alpha=0.5
        )
        expected_half_scale = (amax_scale * weighted_scale).sqrt()
        self.assertTrue(
            bool(torch.allclose(half_scale, expected_half_scale, rtol=1e-6, atol=0))
        )
        self.assertEqual(half_data.dtype, torch.uint8)
        for invalid in (-0.1, 1.1, float("nan")):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                quantize_tensor_rows_activation_mse(
                    values, importance, blend_alpha=invalid
                )


def metadata_target_count(output: Path) -> int:
    return json.loads((output / METADATA_FILENAME).read_text())["quantized_layer_count"]


if __name__ == "__main__":
    unittest.main()
