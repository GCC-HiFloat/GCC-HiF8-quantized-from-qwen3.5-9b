#pragma once

#include <cstdint>

struct Hif8StaticQuantTilingData {
    uint32_t rows;
    uint32_t k;
    uint32_t scaleCount;
};
