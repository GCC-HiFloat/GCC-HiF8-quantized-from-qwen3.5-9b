import math
import unittest
from types import SimpleNamespace

import torch

from vllm_ascend_hif8.calibration import (
    CALIBRATION_FORMAT,
    CALIBRATION_SCHEMA_VERSION,
    DeviceSampleObserver,
    OutputAwareDeviceSampleObserver,
    ScaleSearchConfig,
    qwen35_full_attention_layers,
    qwen35_linear_group_specs,
    sample_token_windows,
    saturating_encode_tensor,
    search_exact_output_aware_static_scale,
    search_output_aware_static_scale,
    search_static_scale,
    static_fake_quantize,
    static_quantize_tensor,
    validate_calibration_artifact,
)
from vllm_ascend_hif8.codec import ASCEND_NAN, ASCEND_NINF, ASCEND_PINF, MAX_FINITE, decode_ascend
from vllm_ascend_hif8.qwen35_calibrator import (
    resolve_runtime_layer_prefix,
    tokenize_wikitext2,
)


class StaticQuantizationTest(unittest.TestCase):
    def test_saturates_finite_outliers_but_preserves_special_values(self):
        source = torch.tensor(
            [-1.0e30, -MAX_FINITE, 0.0, MAX_FINITE, 1.0e30, -math.inf, math.inf, math.nan],
            dtype=torch.float32,
        )
        data = saturating_encode_tensor(source)
        self.assertEqual(data[0].item(), data[1].item())
        self.assertEqual(data[3].item(), data[4].item())
        self.assertTrue(math.isfinite(decode_ascend(data[0].item())))
        self.assertTrue(math.isfinite(decode_ascend(data[4].item())))
        self.assertEqual(data[5].item(), ASCEND_NINF)
        self.assertEqual(data[6].item(), ASCEND_PINF)
        self.assertEqual(data[7].item(), ASCEND_NAN)

    def test_static_scalar_and_group_scales_round_trip(self):
        source = torch.tensor(
            [[[0.25, -0.5], [2.0, -4.0]], [[1.0, -2.0], [8.0, -16.0]]],
            dtype=torch.float32,
        )
        scalar = torch.tensor([2.0], dtype=torch.float32)
        data, returned = static_quantize_tensor(source, scalar)
        expected = torch.tensor([decode_ascend(byte) for byte in data.reshape(-1).tolist()]).reshape(source.shape) * 2.0
        self.assertTrue(torch.equal(returned, scalar))
        self.assertTrue(torch.equal(static_fake_quantize(source, scalar), expected))

        per_head = torch.tensor([1.0, 4.0], dtype=torch.float32)
        grouped = static_fake_quantize(source, per_head, group_axis=1)
        self.assertEqual(grouped.shape, source.shape)
        self.assertTrue(torch.allclose(grouped[:, 0], static_fake_quantize(source[:, 0], per_head[:1])))
        self.assertTrue(torch.allclose(grouped[:, 1], static_fake_quantize(source[:, 1], per_head[1:])))

    def test_rejects_invalid_scale_shapes_and_values(self):
        source = torch.ones(2, 3)
        with self.assertRaises(ValueError):
            static_quantize_tensor(source, torch.ones(2))
        with self.assertRaises(ValueError):
            static_quantize_tensor(source, torch.ones(4), group_axis=1)
        with self.assertRaises(ValueError):
            static_quantize_tensor(source, torch.tensor([0.0]))


class ScaleSearchTest(unittest.TestCase):
    def test_search_beats_naive_unscaled_hif8_on_wide_distribution(self):
        generator = torch.Generator().manual_seed(7)
        samples = torch.randn(100_000, generator=generator) * 12.0
        samples[:20] *= 100.0
        config = ScaleSearchConfig(refinement_steps=9)
        result = search_static_scale(samples, config)
        naive = static_fake_quantize(samples, torch.tensor([1.0]))
        naive_mse = (naive - samples).square().mean().item()
        self.assertGreater(result.scale, 0.0)
        self.assertLess(result.mse, naive_mse)
        self.assertGreater(result.cosine_similarity, 0.99)
        self.assertEqual(result.sample_count, samples.numel())

    def test_search_is_deterministic_and_ignores_nonfinite_values(self):
        samples = torch.tensor([0.0, -0.25, 0.5, 2.0, math.nan, math.inf])
        first = search_static_scale(samples)
        second = search_static_scale(samples)
        self.assertEqual(first, second)
        self.assertEqual(first.nonfinite_count, 2)
        self.assertEqual(first.sample_count, 4)
        self.assertTrue(math.isfinite(first.scale))

    def test_all_nonfinite_or_zero_inputs_have_valid_scales(self):
        nonfinite = search_static_scale(torch.tensor([math.nan, math.inf]))
        zero = search_static_scale(torch.zeros(32))
        self.assertGreater(nonfinite.scale, 0.0)
        self.assertEqual(nonfinite.nonfinite_count, 2)
        self.assertGreater(zero.scale, 0.0)
        self.assertEqual(zero.mse, 0.0)

    def test_output_aware_search_improves_weighted_linear_error(self):
        samples = torch.tensor([[0.05, 0.3, 0.3, 0.3]]).repeat(64, 1)
        importance = torch.tensor([10_000.0, 1.0, 1.0, 1.0])
        result = search_output_aware_static_scale(samples, importance)
        self.assertGreater(result.scale, 0.0)
        self.assertEqual(result.channel_count, 4)
        self.assertLessEqual(
            result.output_weighted_normalized_mse,
            result.baseline_output_weighted_normalized_mse,
        )
        self.assertGreater(result.relative_output_error_improvement, 0.0)

    def test_exact_output_search_keeps_or_improves_reference(self):
        samples = torch.tensor(
            [[0.05, 0.3], [0.05, -0.3], [-0.05, 0.3], [-0.05, -0.3]],
            dtype=torch.float32,
        )
        weights = (
            torch.tensor([[8.0, 1.0], [-8.0, 1.0]], dtype=torch.bfloat16),
        )
        result = search_exact_output_aware_static_scale(
            samples,
            weights,
            reference_scale=0.1,
            diagonal_hint_scale=0.005,
        )
        self.assertLessEqual(
            result.output_weighted_normalized_mse,
            result.reference_output_weighted_normalized_mse,
        )
        self.assertGreaterEqual(result.relative_output_error_improvement, 0.0)
        self.assertEqual(result.output_channel_count, 2)


class ObserverTest(unittest.TestCase):
    def test_bounded_deterministic_scalar_collection(self):
        observer = DeviceSampleObserver(capacity=7, samples_per_observation=4, seed=3)
        observer.observe(torch.arange(20, dtype=torch.float32))
        observer.observe(torch.arange(20, 40, dtype=torch.float32))
        self.assertEqual(observer.sample_count, 7)
        self.assertEqual(observer.cpu_samples().shape, (1, 7))
        summary = observer.summary()
        self.assertEqual(summary["observation_count"], 2)
        self.assertEqual(summary["total_values_per_group"], 40)
        self.assertEqual(summary["max_abs_per_group"], [39.0])

        duplicate = DeviceSampleObserver(capacity=7, samples_per_observation=4, seed=3)
        duplicate.observe(torch.arange(20, dtype=torch.float32))
        duplicate.observe(torch.arange(20, 40, dtype=torch.float32))
        self.assertTrue(torch.equal(observer.cpu_samples(), duplicate.cpu_samples()))

    def test_collects_independent_head_groups(self):
        values = torch.arange(2 * 3 * 5, dtype=torch.float32).reshape(2, 3, 5)
        observer = DeviceSampleObserver(
            capacity=5,
            samples_per_observation=5,
            group_count=3,
            group_axis=1,
        )
        observer.observe(values)
        samples = observer.cpu_samples()
        self.assertEqual(samples.shape, (3, 5))
        self.assertEqual(observer.summary()["max_abs_per_group"], [19.0, 24.0, 29.0])

    def test_output_aware_observer_preserves_complete_rows(self):
        values = torch.arange(6 * 4, dtype=torch.float32).reshape(2, 3, 4)
        observer = OutputAwareDeviceSampleObserver(
            capacity=3, samples_per_observation=2, seed=5
        )
        observer.observe(values)
        observer.observe(values + 100)
        samples = observer.cpu_samples()
        self.assertEqual(samples.shape, (3, 4))
        self.assertTrue(torch.equal(samples[:, 1:] - samples[:, :-1], torch.ones(3, 3)))
        duplicate = OutputAwareDeviceSampleObserver(
            capacity=3, samples_per_observation=2, seed=5
        )
        duplicate.observe(values)
        duplicate.observe(values + 100)
        self.assertTrue(torch.equal(samples, duplicate.cpu_samples()))
        summary = observer.summary()
        self.assertEqual(summary["sample_count_rows"], 3)
        self.assertEqual(summary["input_channel_count"], 4)
        self.assertEqual(summary["total_rows"], 12)
        self.assertEqual(len(summary["channel_mean_square"]), 4)
        self.assertTrue(
            torch.allclose(
                torch.tensor(summary["channel_mean_square"]),
                samples.square().mean(dim=0),
            )
        )


class QwenCalibrationContractTest(unittest.TestCase):
    def test_released_layer_pattern_has_128_linear_groups_and_8_kv_layers(self):
        layer_types = ["linear_attention"] * 32
        for index in range(3, 32, 4):
            layer_types[index] = "full_attention"
        specs = qwen35_linear_group_specs(layer_types)
        self.assertEqual(len(specs), 128)
        self.assertEqual(qwen35_full_attention_layers(layer_types), tuple(range(3, 32, 4)))
        self.assertEqual(len({spec.hook_module for spec in specs}), len(specs))
        self.assertEqual(len({spec.canonical_checkpoint_prefix for spec in specs}), len(specs))

        qkv = next(spec for spec in specs if spec.hook_module.endswith("3.self_attn.q_proj"))
        self.assertEqual(len(qkv.members), 3)
        self.assertTrue(qkv.canonical_checkpoint_prefix.endswith("3.self_attn.q_proj"))

    def test_token_windows_are_unique_and_reproducible(self):
        tokens = torch.arange(100)
        first, starts = sample_token_windows(tokens, sample_count=8, sequence_length=16, seed=42)
        second, starts_again = sample_token_windows(tokens, sample_count=8, sequence_length=16, seed=42)
        self.assertEqual(starts, starts_again)
        self.assertEqual(len(set(starts)), 8)
        self.assertTrue(torch.equal(first, second))
        for row, start in zip(first, starts, strict=True):
            self.assertTrue(torch.equal(row, tokens[start : start + 16]))

    def test_tokenizer_accepts_lazy_dataset_columns(self):
        class Column:
            def __init__(self, values):
                self.values = values

            def __len__(self):
                return len(self.values)

            def __iter__(self):
                return iter(self.values)

        class Dataset:
            def __getitem__(self, name):
                self.assert_name = name
                return Column(["alpha", "beta"])

        class Tokenizer:
            def __call__(self, text, **kwargs):
                self.text = text
                self.kwargs = kwargs
                return {"input_ids": torch.tensor([[1, 2, 3]])}

        dataset = Dataset()
        tokenizer = Tokenizer()
        actual = tokenize_wikitext2(dataset, tokenizer)
        self.assertTrue(torch.equal(actual, torch.tensor([1, 2, 3])))
        self.assertEqual(tokenizer.text, "alpha\n\nbeta")
        self.assertFalse(tokenizer.kwargs["add_special_tokens"])

    def test_resolves_text_only_and_multimodal_hf_module_trees(self):
        class FakeModel:
            def __init__(self, names):
                self.names = names

            def named_modules(self):
                return ((name, SimpleNamespace()) for name in self.names)

        for prefix in ("model.layers", "model.language_model.layers"):
            model = FakeModel(
                {prefix, *(f"{prefix}.{index}" for index in range(4))}
            )
            self.assertEqual(resolve_runtime_layer_prefix(model, 4), prefix)

        incomplete = FakeModel({"model.layers", "model.layers.0"})
        with self.assertRaisesRegex(ValueError, "one complete"):
            resolve_runtime_layer_prefix(incomplete, 4)

    def test_artifact_validation_rejects_missing_or_invalid_scales(self):
        valid = {
            "schema_version": CALIBRATION_SCHEMA_VERSION,
            "format": CALIBRATION_FORMAT,
            "linear_groups": {"group": {"scale": 0.5}},
            "kv_cache": {"layers": {"3": {"key": {"scales": [0.5, 0.6]}, "value": {"scales": [0.7, 0.8]}}}},
        }
        validate_calibration_artifact(valid)
        broken = {**valid, "linear_groups": {"group": {"scale": 0.0}}}
        with self.assertRaises(ValueError):
            validate_calibration_artifact(broken)


if __name__ == "__main__":
    unittest.main()
