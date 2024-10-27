#include "LSlab.h"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cub/util_ptx.cuh>
#include "OperationsDevice.h"
#include <culog/culog.h>
#include <cassert>

#pragma once

namespace lslab {

// unsafe
template<typename K, typename V>
LSLAB_DEVICE size_t
warp_operation_bucketSize(volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, unsigned bucket) {

    using KSub = typename SlabData<K, V>::KSub;
    volatile SlabData<K, V>* next = slabs[bucket];
    size_t count = 0;

    do {
        K read_key = *reinterpret_cast<volatile K*>(next->AddressKey());
        bool isNotEmpty = !(read_key == K{}) && !next->IsDeleted();
        int ballot = __ballot_sync(~0x0, isNotEmpty) & VALID_KEY_MASK;
        next = next->ReadNext();
        assert(__popc(ballot) <= 31);
        count += __popc(ballot);
    } while(next != 0);
    return count;
}

struct NoopConversion {

    template<typename T>
    LSLAB_DEVICE T operator()(const T& x) {
        return x; 
    }

};

// unsafe
template<typename K, typename V, typename VOut, typename Fn>
LSLAB_DEVICE void
warp_operation_readBucket(volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, unsigned bucket, size_t count, K* keys, VOut* values) {

    using KSub = typename SlabData<K, V>::KSub;
    
    const unsigned laneId = threadIdx.x & 0x1Fu;
    auto next = slabs[bucket];
    
    const unsigned threadMask = ~0x0u << (laneId + 1); // all bits but last laneId + 1 set

    size_t i = 0;
    while(i < count) {
        K read_key = *reinterpret_cast<volatile K*>(next->AddressKey());
        bool isNotEmpty = !(read_key == K{}) && !next->IsDeleted();
        int ballot = __ballot_sync(~0x0, isNotEmpty) & VALID_KEY_MASK;
        int before = __popc(ballot & threadMask);
        if(laneId < 31 && isNotEmpty) {
            keys[i + before] = read_key;
            V val = *next->AddressValue();
            values[i + before] = Fn{}(val);
        }
        next = next->ReadNext();
        i += __popc(ballot);
    }

}

}
