#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include "gpumemory.h"
#include <iostream>
#include "OperationsDevice.h"
#include <GroupAllocator/groupallocator>

#pragma once

namespace lslab {

template<typename K, typename V>
std::ostream &operator<<(std::ostream &output, const SlabData<K, V> &s) {
    output << s.keyValue;
    return output;
}

template<typename K, typename V>
WarpAllocCtx<K, V>
setupWarpAllocCtxGroup(groupallocator::GroupAllocator &gAlloc, int threadsPerBlock, int blocks, int gpuid = 0,
                       cudaStream_t stream = cudaStreamDefault) {
    return WarpAllocCtx<K, V>{gAlloc, threadsPerBlock, blocks, gpuid, stream};
}

template<typename K, typename V>
LSLAB_HOST SlabCtx<K, V> *setUpGroup(groupallocator::GroupAllocator &gAlloc, unsigned size, int gpuid = 0,
                          cudaStream_t stream = cudaStreamDefault) {

    //gpuErrchk(cudaSetDevice(gpuid));

    auto sctx = new SlabCtx<K, V>(gAlloc, size, gpuid, stream);
    //sctx->num_of_buckets = size;
    //std::cerr << "Size of index is " << size << std::endl;
    //std::cerr << "Each slab is " << sizeof(SlabData<K, V>) << "B" << std::endl;


    //gAlloc.allocate(&(sctx->slabs), sizeof(void *) * sctx->num_of_buckets, false);

    //for (int i = 0; i < sctx->num_of_buckets; i++) {
    //    gAlloc.allocate(&(sctx->slabs[i]), sizeof(SlabData<K, V>), false);

    //    static_assert(sizeof(sctx->slabs[i]->key[0]) >= sizeof(void *),
    //                  "The key size needs to be greater or equal to the size of a memory address");

    //    //gAlloc.allocate((unsigned long long **) &(sctx->slabs[i][k].keyValue), sizeof(unsigned long long) * 32, false);

    //    memset((void *) (sctx->slabs[i]), 0, sizeof(SlabData<K, V>));

    //    for (int j = 0; j < 31; j++) {
    //        const_cast<typename SlabData<K, V>::KSub *>(sctx->slabs[i]->key)[j] = K{};// EMPTY_PAIR;
    //    }

    //    void **ptrs = (void **) sctx->slabs[i]->key;

    //    ptrs[31] = nullptr;// EMPTY_POINTER;

    //    for (int j = 0; j < 32; j++) {
    //        const_cast<V&>(sctx->slabs[i]->value[j]) = V{};
    //        const_cast<uint64_t&>(sctx->slabs[i]->version[j]) = 0;
    //    }

    //}

    //gAlloc.moveToDevice(gpuid, stream);

    //gpuErrchk(cudaDeviceSynchronize())

    //std::cerr << "Size allocated for Slab: "
    //          << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
    //          << std::endl;
    return sctx;
}

}
