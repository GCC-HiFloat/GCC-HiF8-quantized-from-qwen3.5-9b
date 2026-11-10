#pragma once

#include <cstdint>

struct SoftHif8GemvTilingData {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t hasBias;
    uint32_t biasDtype;
};
