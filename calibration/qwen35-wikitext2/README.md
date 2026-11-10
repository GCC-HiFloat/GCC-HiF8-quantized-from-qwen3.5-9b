# Qwen3.5 static-scale calibration

This directory contains the fixed calibration inputs for the Qwen3.5-9B
static W8A8KV8 checkpoint used on 910B.

## Calibration input and artifact

- `wikitext-2-raw-v1-train.parquet` is the exact cached WikiText-2 raw train
  parquet used during calibration.
- `qwen35-9b-wikitext2-static-calibration.json` contains the searched Linear
  activation scales and the eight observed K/V layer scale groups. It records
  the model identity hashes, dataset revision, dataset SHA256, token-window
  seed/starts, observer settings, and scale-search configuration.
- The parquet SHA256 is
  `e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7`.
- The calibration artifact SHA256 is
  `3d9fa95e435fe2027a5c889f9b518a62ecf47e4a75800a5c03ded231efa463cf`.

The calibration implementation is maintained in:

- `src/vllm_ascend_hif8/calibration.py`
- `src/vllm_ascend_hif8/qwen35_calibrator.py`
- `scripts/calibrate_qwen35_static_hif8.py`

Run calibration with the local parquet:

```bash
python scripts/calibrate_qwen35_static_hif8.py \
  /path/to/Qwen3.5-9B \
  /path/to/qwen35-9b-wikitext2-static-calibration.json \
  --dataset-parquet /path/to/wikitext-2-raw-v1-train.parquet \
  --dataset-revision b08601e04326c79dfdd32d625aee71d232d685c3 \
  --samples 128 --sequence-length 2048 --seed 42
```

## Evaluation

The train split supplies offline calibration inputs. ACC evaluation uses
separate task splits; see [Accuracy evaluation](../../docs/validation-status.md).
