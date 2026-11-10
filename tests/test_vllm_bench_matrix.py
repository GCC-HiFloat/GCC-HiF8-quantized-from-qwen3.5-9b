"""Test matrix validation and report generation without an NPU."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import publish_vllm_bench_matrix as publisher
import run_vllm_bench_matrix as runner


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "matrix"
        self.root.mkdir()
        self.manifest = {
            "profiles": runner.PROFILES, "num_prompts": 16, "num_warmups": 4,
            "repetitions": 3, "max_model_len": 8192, "kv_bytes": 1073741824, "card": 6,
            "models": {"bf16": "/model/bf16", "hif8": "/model/hif8"},
            "source_head": "test-head", "tracked_diff_sha256": "test-diff",
        }
        (self.root / "matrix.json").write_text(json.dumps(self.manifest))

    def record(self, path):
        profile, trial, name = path.relative_to(self.root).parts
        mode, batch = name.split("-bs")
        length, output = runner.PROFILES[profile]
        speed = (10 if mode == "bf16" else 5) + int(trial[-1]) - 2
        return {
            "protocol": {"random-input-len": length, "random-output-len": output,
                         "num-prompts": 16, "num-warmups": 4, "max-model-len": 8192,
                         "max-num-seqs": int(batch), "max-num-batched-tokens": 8192,
                         "kv-cache-memory-bytes": self.manifest["kv_bytes"], "tensor-parallel-size": 1, "seed": 42},
            "command": {"argv": ["--model", f"/model/{mode}"], "npu_smi_card": 6,
                        "environment": {"HIF8_STATIC_FUSED_GENERATION": "1", "HIF8_KV_CACHE": "1",
                                        "HIF8_COMPUTE_BACKEND": "soft_hif8"}},
            "log_excerpts": ["Loaded checkpoint-static", "KV writes use static_quant"],
            "summary": {"output_tokens_per_second": speed, "hbm_sampled_peak_mb": 20000,
                        "hbm_baseline_mb": 3000, "official_throughput": {"requests_per_second": speed / output}},
        }

    def test_jobs_cover_all_pairs_and_alternate_order(self):
        jobs = list(runner.jobs(self.manifest))
        self.assertEqual(len(jobs), 36)
        self.assertEqual(len(set(jobs)), 36)
        self.assertEqual([job[-1] for job in jobs[:4]], ["bf16", "hif8", "hif8", "bf16"])

    def test_aggregation_uses_medians_and_ranges(self):
        with patch.object(publisher, "collect", side_effect=self.record):
            _, evidence, rows = publisher.aggregate(self.root)
        self.assertEqual(len(evidence), 36)
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["hif8_bf16_ratio"], .5)
        self.assertEqual(rows[0]["hif8"]["min_output_tokens_per_second"], 4)
        self.assertEqual(rows[0]["hif8"]["max_output_tokens_per_second"], 6)

    def test_missing_measurement_stops_publication(self):
        with patch.object(publisher, "collect", side_effect=FileNotFoundError("unfinished")):
            with self.assertRaises(FileNotFoundError):
                publisher.aggregate(self.root)

    def test_resume_rejects_different_request_count(self):
        record = self.record(self.root / "interactive/repeat-1/bf16-bs4")
        record["protocol"]["num-prompts"] = 64
        with self.assertRaises(ValueError):
            runner.validate_run(record, self.manifest, "interactive", 4, "bf16")

    def test_report_preserves_accuracy_section(self):
        self.manifest["kv_bytes"] = 2147483648
        (self.root / "matrix.json").write_text(json.dumps(self.manifest))
        with patch.object(publisher, "collect", side_effect=self.record):
            manifest, evidence, rows = publisher.aggregate(self.root)
        repo = Path(self.temp.name) / "repo"
        (repo / "docs").mkdir(parents=True)
        (repo / "README.md").write_text("# Test\n\n## 精度\nunchanged\n\n## 性能\nold\n\n## 代码结构\nunchanged\n")
        paths = publisher.write_reports(repo, self.root, manifest, evidence, rows)
        readme = (repo / "README.md").read_text()
        self.assertIn("## 精度\nunchanged", readme)
        self.assertIn("## 代码结构\nunchanged", readme)
        self.assertIn("512/1024", readme)
        self.assertIn("KV 预算 2 GiB", readme)
        report = (repo / "docs/vllm-bench-report.md").read_text()
        self.assertIn("| KV 预算 | 2 GiB |", report)
        self.assertIn("## 重复测量波动", report)
        self.assertIn("3,000～3,000 MB", report)
        self.assertIn("benchmarking.md", report)
        self.assertNotIn("## 性能\nold", readme)
        self.assertEqual(len(paths), 3)
        for name in ("evidence.json", "aggregate.json", "provenance.json", "matrix.json"):
            self.assertTrue((repo / "docs/reports/vllm-bench" / name).is_file())
        self.assertFalse((repo / "docs/performance-summary.md").exists())
        self.assertEqual(list((repo / "docs").rglob("*.md")), [repo / "docs/vllm-bench-report.md"])
        # A later publication updates the same report and evidence directory.
        publisher.write_reports(repo, self.root, manifest, evidence, rows)
        self.assertEqual(len(list((repo / "docs/reports").iterdir())), 1)

    def test_cache_sufficient_matrix_requires_zero_preemptions(self):
        self.manifest["require_no_preemptions"] = True
        record = self.record(self.root / "interactive/repeat-1/bf16-bs4")
        with self.assertRaisesRegex(ValueError, "Missing scheduler"):
            runner.validate_run(record, self.manifest, "interactive", 4, "bf16")
        record["summary"].update(preemption_tracking=True, preemptions_including_warmup=7)
        with self.assertRaisesRegex(ValueError, "Cache preemptions"):
            runner.validate_run(record, self.manifest, "interactive", 4, "bf16")
        record["summary"]["preemptions_including_warmup"] = 0
        runner.validate_run(record, self.manifest, "interactive", 4, "bf16")


if __name__ == "__main__":
    unittest.main()
