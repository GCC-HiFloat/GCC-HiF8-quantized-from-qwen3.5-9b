"""Validate publishing without importing torch or running a model."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "publisher", Path(__file__).resolve().parents[1] / "scripts/summarize_vllm_bench.py")
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.summary = {
            "returncode": 0, "output_tokens": 256, "valid_memory_sample_count": 1,
            "hbm_baseline_mb": 3000, "hbm_sampled_peak_mb": 3000,
            "output_tokens_per_second": 25.6,
            "official_throughput": {"num_requests": 8, "total_num_tokens": 1280, "elapsed_time": 10.0},
        }
        self.argv = ["vllm", "bench", "throughput", "--model", str(self.root),
                     "--random-input-len", "128", "--random-output-len", "32",
                     "--num-prompts", "8", "--num-warmups", "4", "--max-model-len", "512",
                     "--max-num-seqs", "4", "--max-num-batched-tokens", "8192",
                     "--kv-cache-memory-bytes", "1073741824", "--tensor-parallel-size", "1",
                     "--seed", "42", "--dataset-name", "random", "--random-range-ratio", "0",
                     "--dtype", "bfloat16", "--backend", "vllm", "--enforce-eager",
                     "--no-async-scheduling", "--no-enable-prefix-caching"]
        (self.root / "command.json").write_text(json.dumps({
            "argv": self.argv, "started_unix": 0}))
        (self.root / "config.json").write_text("{}")
        (self.root / "console.log").write_text("Loading model weights took 10.0000 GB\n")
        (self.root / "memory-samples.jsonl").write_text(json.dumps({"hbm": {"used_mb": 3000}}) + "\n")

    def collect(self):
        (self.root / "summary.json").write_text(json.dumps(self.summary))
        (self.root / "throughput.json").write_text(json.dumps(self.summary["official_throughput"]))
        return publisher.collect(self.root)

    def test_keeps_raw_evidence_and_hash(self):
        result = self.collect()
        self.assertEqual(result["weight_load_log_value_gb"], 10)
        self.assertEqual(result["memory_evidence"]["sampled_peak"]["hbm"]["used_mb"], 3000)
        self.assertEqual(len(result["sha256"]["console.log"]), 64)

    def test_rejects_failed_run(self):
        self.summary["returncode"] = 1
        with self.assertRaises(ValueError):
            self.collect()

    def test_rejects_wrong_token_protocol(self):
        self.summary["official_throughput"]["total_num_tokens"] = 9216
        with self.assertRaises(ValueError):
            self.collect()

    def test_rejects_missing_memory(self):
        self.summary["valid_memory_sample_count"] = 0
        with self.assertRaises(ValueError):
            self.collect()

    def test_accepts_longer_fixed_workload(self):
        command = json.loads((self.root / "command.json").read_text())
        for flag, value in (("--random-input-len", "512"), ("--random-output-len", "1024"),
                            ("--num-prompts", "64"), ("--max-model-len", "8192")):
            command["argv"][command["argv"].index(flag) + 1] = value
        (self.root / "command.json").write_text(json.dumps(command))
        self.summary["official_throughput"].update(num_requests=64, total_num_tokens=64 * 1536)
        self.summary.update(output_tokens=65536, output_tokens_per_second=6553.6)
        self.assertEqual(self.collect()["protocol"]["random-output-len"], 1024)

    def test_rejects_early_eos(self):
        self.summary["output_tokens"] = 255
        with self.assertRaises(ValueError):
            self.collect()

    def test_rejects_incorrect_throughput(self):
        self.summary["output_tokens_per_second"] = 100
        with self.assertRaises(ValueError):
            self.collect()

    def test_preemption_count_matches_log(self):
        self.summary.update(preemption_tracking=True, preemptions_including_warmup=0)
        with self.assertRaisesRegex(ValueError, "preemption count"):
            self.collect()
        (self.root / "console.log").write_text(
            "HIF8_BENCH_PREEMPTION_TRACKING=1\nHIF8_BENCH_PREEMPTION=1\n")
        with self.assertRaisesRegex(ValueError, "preemption count"):
            self.collect()
        self.summary["preemptions_including_warmup"] = 1
        self.assertEqual(self.collect()["summary"]["preemptions_including_warmup"], 1)

    def test_rejects_mismatched_pair(self):
        bf16 = self.collect()
        hif8 = self.collect()
        hif8["protocol"]["random-input-len"] = 512
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            publisher.validate_pair(bf16, hif8)


if __name__ == "__main__":
    unittest.main()
