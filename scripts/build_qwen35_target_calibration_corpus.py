#!/usr/bin/env python3
"""Build a disjoint multiple-choice calibration corpus for Qwen3.5 static HIF8.

The resulting parquet intentionally contains only prompt text (never answer
labels).  It is used by ``calibrate_qwen35_static_hif8.py`` to search fixed
activation/KV scales on text distributions close to the evaluation workload,
without calibrating on the evaluation split itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from pathlib import Path


def _read_parquet(path: Path):
    from datasets import load_dataset

    return load_dataset("parquet", data_files={"data": str(path)}, split="data")


def _read_csv(path: Path):
    from datasets import load_dataset

    return load_dataset("csv", data_files={"data": str(path)}, split="data")


def _arc_rows(path: Path) -> list[str]:
    rows = []
    for item in _read_parquet(path):
        choices = item["choices"]["text"]
        rows.append(
            "Question: "
            + str(item["question"])
            + "\n"
            + "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(choices))
            + "\nAnswer:"
        )
    return rows


def _piqa_rows(path: Path) -> list[str]:
    rows = []
    with zipfile.ZipFile(path) as archive:
        name = next(
            name
            for name in archive.namelist()
            if name.endswith("/train.jsonl") or name.endswith("/dev.jsonl")
        )
        with archive.open(name) as stream:
            for raw in stream:
                item = json.loads(raw.decode("utf-8"))
                rows.append(
                    f"Question: {item['goal']}\nA. {item['sol1']}\nB. {item['sol2']}\nAnswer:"
                )
    return rows


def _winogrande_rows(path: Path) -> list[str]:
    rows = []
    for item in _read_parquet(path):
        rows.append(
            f"Question: {item['sentence']}\nA. {item['option1']}\nB. {item['option2']}\nAnswer:"
        )
    return rows


def _boolq_rows(path: Path) -> list[str]:
    rows = []
    for item in _read_parquet(path):
        rows.append(
            f"{item['passage']}\nQuestion: {item['question']}?\nA. no\nB. yes\nAnswer:"
        )
    return rows


def _mmlu_pro_rows(path: Path) -> list[str]:
    rows = []
    for item in _read_parquet(path):
        options = item["options"]
        rows.append(
            f"Question: {item['question']}\n"
            + "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(options))
            + "\nAnswer:"
        )
    return rows


def _mmlupro_x_rows(path: Path) -> list[str]:
    rows = []
    for item in _read_parquet(path):
        options = [item[f"option_{i}"] for i in range(10)]
        rows.append(
            f"Question: {item['question']}\n"
            + "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(options))
            + "\nAnswer:"
        )
    return rows


def _cmmlu_dev_rows(root: Path) -> list[str]:
    rows = []
    for path in sorted((root / "dev").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as stream:
            for item in csv.DictReader(stream):
                rows.append(
                    f"{item['Question']}\n"
                    f"A. {item['A']}\nB. {item['B']}\nC. {item['C']}\nD. {item['D']}\n答案："
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--arc", type=Path, required=True)
    parser.add_argument("--piqa-zip", type=Path, required=True)
    parser.add_argument("--winogrande", type=Path, required=True)
    parser.add_argument("--boolq", type=Path, required=True)
    parser.add_argument("--mmlu-pro", type=Path, required=True)
    parser.add_argument("--mmlu-prox-en", type=Path, required=True)
    parser.add_argument("--cmmlu-dev", type=Path, required=True)
    parser.add_argument("--per-source", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if args.per_source < 1:
        raise ValueError("--per-source must be positive")

    groups = {
        "arc_easy_train": _arc_rows(args.arc),
        "piqa_train": _piqa_rows(args.piqa_zip),
        "winogrande_train": _winogrande_rows(args.winogrande),
        "boolq_train": _boolq_rows(args.boolq),
        "mmlu_pro_validation": _mmlu_pro_rows(args.mmlu_pro),
        "mmlu_prox_en_validation": _mmlupro_x_rows(args.mmlu_prox_en),
        "cmmlu_dev": _cmmlu_dev_rows(args.cmmlu_dev),
    }
    rng = random.Random(args.seed)
    selected: list[str] = []
    provenance: dict[str, int] = {}
    for name, rows in groups.items():
        rows = list(rows)
        rng.shuffle(rows)
        rows = rows[: args.per_source]
        provenance[name] = len(rows)
        selected.extend(rows)
    rng.shuffle(selected)
    if not selected:
        raise RuntimeError("calibration corpus is empty")

    import pyarrow as pa
    import pyarrow.parquet as pq

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"text": pa.array(selected, type=pa.string())})
    metadata = dict(table.schema.metadata or {})
    metadata[b"qwen35_calibration_sources"] = json.dumps(
        provenance, sort_keys=True
    ).encode("utf-8")
    metadata[b"qwen35_calibration_seed"] = str(args.seed).encode("ascii")
    table = table.replace_schema_metadata(metadata)
    temporary = args.output.with_name(args.output.name + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(args.output)
    print(json.dumps({"rows": len(selected), "sources": provenance}, sort_keys=True))


if __name__ == "__main__":
    main()
