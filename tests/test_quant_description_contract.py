"""Contract tests for the vLLM / vllm-ascend quantization lookup path.

These reproduce the upstream lookup rules against the exporter's real output
naming so a rename on either side fails here instead of at serving time:

* vLLM applies ``hf_to_vllm_mapper.get_unstacked_mapper()`` to the ModelSlim
  ``quant_description``. For Qwen3.5 that mapper only rewrites prefixes
  (``model.language_model.`` -> ``language_model.model.``); the stacked entries
  that fuse ``in_proj_qkv``/``in_proj_z`` into ``in_proj_qkvz`` are dropped, so
  the per-component checkpoint names must survive.
* vllm-ascend then resolves a fused runtime linear by looking up every shard
  listed in ``packed_modules_model_mapping["qwen3_5"]`` and requires all shards
  of one layer to carry the same quant type.
"""

import unittest

from vllm_ascend_hif8.exporter import (
    DYNAMIC_QUANT_TYPE,
    QUANT_TYPE,
    STATIC_QUANT_TYPE,
    build_quant_description,
)

# vllm/model_executor/models/qwen3_vl.py: Qwen3VLForConditionalGeneration
UNSTACKED_PREFIX_MAP = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
}

# vllm_ascend/quantization/modelslim_config.py: packed_modules_model_mapping
QWEN3_5_PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def apply_unstacked_prefixes(name: str) -> str:
    """Mirror ``WeightsMapper._map_name`` for prefix-only mappings."""
    for prefix, replacement in UNSTACKED_PREFIX_MAP.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix) :]
    return name


def get_linear_quant_type(quant_description, prefix, packed_modules_mapping):
    """Mirror ``vllm_ascend.quantization.modelslim_config.get_linear_quant_type``."""
    proj_name = prefix.split(".")[-1]
    if proj_name not in packed_modules_mapping:
        return quant_description[prefix + ".weight"]

    quant_type = None
    for shard_proj_name in packed_modules_mapping[proj_name]:
        shard_quant_type = quant_description[prefix.replace(proj_name, shard_proj_name) + ".weight"]
        if quant_type is None:
            quant_type = shard_quant_type
        elif shard_quant_type != quant_type:
            raise ValueError(f"Not all shards of {prefix} are quantized with same quant type")
    return quant_type


def qwen35_checkpoint_names(layer_types):
    names = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
    for index, kind in enumerate(layer_types):
        prefix = f"model.language_model.layers.{index}"
        names += [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.mlp.down_proj.weight",
        ]
        if kind == "full_attention":
            names += [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.o_proj.weight",
                f"{prefix}.self_attn.q_norm.weight",
                f"{prefix}.self_attn.k_norm.weight",
            ]
        else:
            names += [
                f"{prefix}.linear_attn.in_proj_qkv.weight",
                f"{prefix}.linear_attn.in_proj_z.weight",
                f"{prefix}.linear_attn.in_proj_b.weight",
                f"{prefix}.linear_attn.in_proj_a.weight",
                f"{prefix}.linear_attn.out_proj.weight",
                f"{prefix}.linear_attn.conv1d.weight",
                f"{prefix}.linear_attn.norm.weight",
            ]
    names += ["model.visual.blocks.0.attn.qkv.weight", "model.visual.merger.linear_fc1.weight"]
    return names


class QuantDescriptionLookupTest(unittest.TestCase):
    LAYER_TYPES = ("linear_attention", "linear_attention", "linear_attention", "full_attention")

    def setUp(self):
        description = build_quant_description(tuple(qwen35_checkpoint_names(self.LAYER_TYPES)))
        self.mapped = {apply_unstacked_prefixes(name): kind for name, kind in description.items()}

    def test_unstacked_mapper_moves_the_description_onto_runtime_prefixes(self):
        self.assertIn("language_model.model.layers.0.mlp.down_proj.weight", self.mapped)
        self.assertIn("language_model.model.layers.3.self_attn.q_proj.weight", self.mapped)
        self.assertNotIn("model.language_model.layers.0.mlp.down_proj.weight", self.mapped)
        self.assertEqual(self.mapped["language_model.model.embed_tokens.weight"], QUANT_TYPE)
        self.assertEqual(self.mapped["language_model.lm_head.weight"], "FLOAT")
        self.assertEqual(self.mapped["visual.blocks.0.attn.qkv.weight"], "FLOAT")

    def test_every_fused_runtime_linear_resolves_to_one_quant_type(self):
        expected = {}
        for index, kind in enumerate(self.LAYER_TYPES):
            prefix = f"language_model.model.layers.{index}"
            expected[f"{prefix}.mlp.gate_up_proj"] = QUANT_TYPE
            expected[f"{prefix}.mlp.down_proj"] = QUANT_TYPE
            if kind == "full_attention":
                expected[f"{prefix}.self_attn.qkv_proj"] = QUANT_TYPE
                expected[f"{prefix}.self_attn.o_proj"] = QUANT_TYPE
            else:
                expected[f"{prefix}.linear_attn.in_proj_qkvz"] = QUANT_TYPE
                expected[f"{prefix}.linear_attn.out_proj"] = QUANT_TYPE
                # in_proj_b and in_proj_a stay floating point, so their fused
                # linear must resolve to FLOAT without a mixed-shard error.
                expected[f"{prefix}.linear_attn.in_proj_ba"] = "FLOAT"

        for prefix, quant_type in expected.items():
            self.assertEqual(get_linear_quant_type(self.mapped, prefix, QWEN3_5_PACKED_MODULES), quant_type, prefix)

    def test_mixed_shards_would_be_rejected_upstream(self):
        mapped = dict(self.mapped)
        mapped["language_model.model.layers.0.linear_attn.in_proj_z.weight"] = "FLOAT"
        with self.assertRaises(ValueError):
            get_linear_quant_type(
                mapped, "language_model.model.layers.0.linear_attn.in_proj_qkvz", QWEN3_5_PACKED_MODULES
            )

    def test_static_description_marks_every_fused_component_static(self):
        description = build_quant_description(
            tuple(qwen35_checkpoint_names(self.LAYER_TYPES)),
            static_activation=True,
        )
        mapped = {
            apply_unstacked_prefixes(name): kind
            for name, kind in description.items()
        }
        self.assertEqual(
            mapped["language_model.model.embed_tokens.weight"],
            DYNAMIC_QUANT_TYPE,
        )
        for prefix in (
            "language_model.model.layers.3.self_attn.qkv_proj",
            "language_model.model.layers.0.mlp.gate_up_proj",
            "language_model.model.layers.0.linear_attn.in_proj_qkvz",
        ):
            self.assertEqual(
                get_linear_quant_type(mapped, prefix, QWEN3_5_PACKED_MODULES),
                STATIC_QUANT_TYPE,
            )

    def test_unquantized_modules_are_present_so_lookup_never_raises_keyerror(self):
        for name in ("input_layernorm", "post_attention_layernorm"):
            self.assertEqual(self.mapped[f"language_model.model.layers.0.{name}.weight"], "FLOAT")
        self.assertEqual(self.mapped["language_model.model.layers.0.linear_attn.conv1d.weight"], "FLOAT")


if __name__ == "__main__":
    unittest.main()
