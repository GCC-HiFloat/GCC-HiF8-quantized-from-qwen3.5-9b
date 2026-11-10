#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
soc=${1:-ascend910b}
case "$soc" in
  ascend910b) ;;
  *)
    printf 'unsupported SoC target: %s\n' "$soc" >&2
    exit 2
    ;;
esac

ascend_home=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}
build_dir=${HIF8_OPP_BUILD_DIR:-"$script_dir/build/$soc"}
decode_probe_stage=${HIF8_DECODE_PROBE_STAGE:-0}
rows_per_chunk=${HIF8_ROWS_PER_CHUNK:-4}
use_int_offset_shift=${HIF8_USE_INT_OFFSET_SHIFT:-0}
decode_impl=${HIF8_DECODE_IMPL:-0}
decode_queue_depth=${HIF8_DECODE_QUEUE_DEPTH:-2}
reuse_decode_output_as_offset=${HIF8_REUSE_DECODE_OUTPUT_AS_OFFSET:-0}
online_bf16_arith=${HIF8_ONLINE_BF16_ARITH:-0}
use_tscm_pipeline=${HIF8_USE_TSCM_PIPELINE:-0}
local_tile_sync=${HIF8_LOCAL_TILE_SYNC:-0}
skip_final_tile_barrier=${HIF8_SKIP_FINAL_TILE_BARRIER:-1}
pairwise_epilogue_sync=${HIF8_PAIRWISE_EPILOGUE_SYNC:-1}
reuse_weight_tiles=${HIF8_REUSE_WEIGHT_TILES:-1}
wide_n_tile=${HIF8_WIDE_N_TILE:-1}
fused_vector_encode=${HIF8_FUSED_VECTOR_ENCODE:-0}

cmake -S "$script_dir" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SOURCE_PACKAGE=OFF \
  -DENABLE_BINARY_PACKAGE=ON \
  -DASCEND_COMPUTE_UNIT="$soc" \
  -Dvendor_name=vllm_ascend_hif8 \
  -DASCEND_CANN_PACKAGE_PATH="$ascend_home" \
  -DASCEND_PYTHON_EXECUTABLE=python3 \
  -DHIF8_DECODE_PROBE_STAGE="$decode_probe_stage" \
  -DHIF8_ROWS_PER_CHUNK="$rows_per_chunk" \
  -DHIF8_USE_INT_OFFSET_SHIFT="$use_int_offset_shift" \
  -DHIF8_DECODE_IMPL="$decode_impl" \
  -DHIF8_DECODE_QUEUE_DEPTH="$decode_queue_depth" \
  -DHIF8_REUSE_DECODE_OUTPUT_AS_OFFSET="$reuse_decode_output_as_offset" \
  -DHIF8_ONLINE_BF16_ARITH="$online_bf16_arith" \
  -DHIF8_USE_TSCM_PIPELINE="$use_tscm_pipeline" \
  -DHIF8_LOCAL_TILE_SYNC="$local_tile_sync" \
  -DHIF8_SKIP_FINAL_TILE_BARRIER="$skip_final_tile_barrier" \
  -DHIF8_PAIRWISE_EPILOGUE_SYNC="$pairwise_epilogue_sync" \
  -DHIF8_REUSE_WEIGHT_TILES="$reuse_weight_tiles" \
  -DHIF8_WIDE_N_TILE="$wide_n_tile" \
  -DHIF8_FUSED_VECTOR_ENCODE="$fused_vector_encode" \
  -DCMAKE_INSTALL_PREFIX="$build_dir" \
  -DENABLE_CROSS_COMPILE=OFF
cmake --build "$build_dir" --target binary package --parallel "${MAX_JOBS:-$(nproc)}"
