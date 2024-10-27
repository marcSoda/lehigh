#include "LSlab.h"
#include <cuda_runtime.h>
#include <cub/util_ptx.cuh>

#pragma once

namespace lslab {
template<typename T>
LSLAB_DEVICE T shfl(unsigned mask, T val, int offset) {
    return cub::ShuffleIndex<32>(val, offset, mask);
}

template<>
LSLAB_DEVICE unsigned shfl(unsigned mask, unsigned val, int offset) {
    return __shfl_sync(mask, val, offset);
}

template<>
LSLAB_DEVICE int shfl(unsigned mask, int val, int offset) {
    return __shfl_sync(mask, val, offset);
}

template<>
LSLAB_DEVICE unsigned long long shfl(unsigned mask, unsigned long long val, int offset) {
    return __shfl_sync(mask, val, offset);
}
}
