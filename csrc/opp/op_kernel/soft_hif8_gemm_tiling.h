#pragma once

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct SoftHif8GemmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t hasBias;
    uint32_t biasDtype;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t outputStride;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileK;
    uint32_t mTileCount;
    uint32_t nTileCount;
    uint32_t kTileCount;
    uint64_t outputWorkspaceOffset;
    uint64_t stagingWorkspaceOffset;
    uint64_t stagingWorkspaceStride;
    uint32_t useActivationCache;
    uint32_t useMultiLineDma;
    uint64_t activationCacheOffset;
    AscendC::tiling::TCubeTiling matmulTiling;
};

struct SoftHif8FusedGemmTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t hasBias;
    uint32_t biasDtype;
    uint32_t aicNum;
    uint32_t aivNum;
    uint32_t outputStride;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t tileK;
    uint32_t mTileCount;
    uint32_t nTileCount;
    uint32_t kTileCount;
    uint64_t outputWorkspaceOffset;
    uint64_t activationWorkspaceOffset;
    uint64_t tokenScaleWorkspaceOffset;
    uint64_t weightWorkspaceOffset;
    uint64_t weightWorkspaceStride;
    AscendC::tiling::TCubeTiling matmulTiling;
};

// Kept as a distinct type so the fixed-scale operator cannot accidentally be
// compiled against the dynamic fused ABI.  The workspace and Cube schedule
// are intentionally identical; only the activation scale source differs.
using SoftHif8StaticFusedGemmTilingData = SoftHif8FusedGemmTilingData;
