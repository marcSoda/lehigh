#include "LSlab.h"
#include <cstdio>
#include <culog/culog.h>
#include "shfl.h"
#include "SlabData.h"

#pragma once

namespace lslab {

template<typename K, typename V>
struct MemoryBlock {
    unsigned long long bitmap;
    SlabData<K, V> *slab;// 64 slabs
};

template<typename K, typename V>
struct SuperBlock {
    MemoryBlock<K, V> *memblocks;// 32 * n memblocks (n for each thread)
};

template<typename K, typename V, uint32_t N = 1>
struct WarpAllocCtx {
    WarpAllocCtx() : blocks(nullptr) {}

    WarpAllocCtx(groupallocator::GroupAllocator &gAlloc, int threadsPerBlock, int numblocks, int gpuid = 0,
                       cudaStream_t stream = cudaStreamDefault) {
        gpuErrchk(cudaSetDevice(gpuid));
        gAlloc.allocate(&blocks, (size_t) ceil(threadsPerBlock * numblocks / 32.0) * sizeof(SuperBlock<K, V>), false);
        for (size_t i = 0; i < (size_t) ceil(threadsPerBlock * numblocks / 32.0); i++) {
            gAlloc.allocate(&(blocks[i].memblocks), sizeof(MemoryBlock<K, V>) * 32 * N, false);
            for (int j = 0; j < 32 * N; j++) {
                blocks[i].memblocks[j].bitmap = ~0ull;
                gAlloc.allocate(&(blocks[i].memblocks[j].slab), sizeof(SlabData<K, V>) * 64, false);
                for (int k = 0; k < 64; k++) {
                    blocks[i].memblocks[j].slab[k].ilock = 0;
                    for (int w = 0; w < 32; w++) {
                        blocks[i].memblocks[j].slab[k].key[w] = K{};
                        blocks[i].memblocks[j].slab[k].value[w] = V{};
                        blocks[i].memblocks[j].slab[k].version[w] = 0;
                    }
                }
            }
        }
        gAlloc.moveToDevice(gpuid, stream);
        gpuErrchk(cudaDeviceSynchronize())
        std::cerr << "Size allocated for warp alloc: "
                  << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
                  << std::endl;

    }

    LSLAB_DEVICE volatile SlabData<K, V>* allocate() {
        // just doing parallel shared-nothing allocation
        const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
        const unsigned laneId = threadIdx.x & 0x1Fu;
        if (this->blocks == nullptr) {
            __trap();
            return 0;
        }
        
        MemoryBlock<K, V> *mblocks; 
        unsigned bitmap;
        int index;
        int ballotThread; 
       
        #pragma unroll 
        for(uint32_t mult = 1; mult <= N; mult++) { 
            mblocks = this->blocks[warpIdx * mult].memblocks;
            bitmap = mblocks[laneId].bitmap;
            index = __ffs((int) bitmap) - 1;
            ballotThread = __ffs((int) __ballot_sync(~0u, (index != -1))) - 1;
            if(ballotThread != -1) goto allocate; 
        }
        if(laneId == 0) {
            printf("Ran out of memory (%u,%u)\n", blockIdx.x, threadIdx.x);
        }
        __threadfence_system();
        __syncwarp();
        __trap();
        return 0;
        allocate:
        auto location = (mblocks[laneId].slab + index);
        assert(location != nullptr);
        if (ballotThread == laneId) {
            bitmap = bitmap ^ (1u << (unsigned) index);
            mblocks[laneId].bitmap = bitmap;
        }
        location = shfl(~0u, location, ballotThread);
    
        CULOGF(laneId == 0, "Allocated: %p thread (%u, %u) line %d\n", location, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), __LINE__);
        assert(location != nullptr);
        return location;
    }
    
    LSLAB_DEVICE void deallocate(volatile SlabData<K, V>* l) {
    
        const unsigned warpIdx = (threadIdx.x / 32) + blockIdx.x * (blockDim.x / 32);
        const unsigned laneId = threadIdx.x & 0x1Fu;
        if (this->blocks == nullptr) {
            return;
        }
    
        for(uint32_t mult = 1; mult <= N; mult++) { 
            MemoryBlock<K, V> *blocks = this->blocks[warpIdx * mult].memblocks;
            if ((unsigned long long) blocks[laneId].slab <= (unsigned long long) l && (unsigned long long) (blocks[laneId].slab + 32) > l) {
                unsigned diff = (unsigned long long) l - (unsigned long long) blocks[laneId].slab;
                unsigned idx = diff / sizeof(SlabData<K, V>);
                blocks[laneId].bitmap = blocks[laneId].bitmap | (1u << idx);
                return;
            }
        }
    }

    SuperBlock<K, V> *blocks;
    // there should be a block per warp ie threadsPerBlock * blocks / 32 superblocks
};

}
