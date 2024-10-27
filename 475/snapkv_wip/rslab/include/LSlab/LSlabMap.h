#include "OperationsDevice.h"
#include "RSlabOperationsDevice.h"

#pragma once

namespace lslab {

template<typename K, typename V>
class LSlab {
public:

    LSLAB_HOST_DEVICE LSlab(volatile SlabData<K, V>** s, unsigned n, WarpAllocCtx<K, V> c) : slabs(s), number_of_buckets(n), ctx(c) {

    }

    LSLAB_HOST_DEVICE LSlab() : slabs(nullptr), number_of_buckets(0), ctx() {
    }

    LSLAB_HOST_DEVICE ~LSlab() {} 

    LSLAB_DEVICE void get(K& key, V& value, unsigned hash, bool threadMask) {
        warp_operation_search(threadMask, key, value, hash, slabs, number_of_buckets);
    }
 
    LSLAB_DEVICE void unsafeGet(K& key, V& value, unsigned hash, bool threadMask) {
        warp_operation_unsafe_search(threadMask, key, value, hash, slabs, number_of_buckets);
    }
   
    LSLAB_DEVICE void put(K& key, V& value, unsigned hash, uint64_t version, bool threadMask) {
        warp_operation_replace(threadMask, key, value, hash, version, slabs, number_of_buckets, ctx);
    }
    
    LSLAB_DEVICE void remove(K& key, V& value, unsigned hash, uint64_t version, bool threadMask) {
        warp_operation_delete(threadMask, key, value, hash, version, slabs, number_of_buckets);
    }

    //LSLAB_DEVICE void modify(K& key, V& value, unsigned hash, Operation op, bool threadMask = false) {
    //    warp_operation_delete_or_replace(threadMask, key, value, hash, slabs, number_of_buckets, ctx, op);
    //}

    LSLAB_DEVICE size_t unsafeBucketSize(unsigned bucket) {
        return warp_operation_bucketSize(slabs, number_of_buckets, bucket);
    }

    template<typename VOut = V, typename Fn = NoopConversion>
    LSLAB_DEVICE size_t unsafeReadBucket(unsigned bucket, K* keys, VOut* values, size_t contains) {
        warp_operation_readBucket<K, V, VOut, Fn>(slabs, number_of_buckets, bucket, contains, keys, values);
    }

    LSLAB_DEVICE void sharedLockBucket(unsigned bucket) {
        slabs[bucket]->SharedLockSlab();
    }

    LSLAB_DEVICE void sharedUnlockBucket(unsigned bucket) {
        slabs[bucket]->SharedUnlockSlab();
    }

    LSLAB_DEVICE void getAll(K* keys, V* values, size_t bufSize, unsigned long long* written, int* fail) {
        
        int wid = (threadIdx.x + blockIdx.x * blockDim.x) / 32;
        int warps = (blockDim.x * gridDim.x) / 32;

        for(int i = wid; i < number_of_buckets; i+= warps) {
            
            sharedLockBucket(i);

            size_t size = unsafeBucketSize(i);

            unsigned long long start = 0;
            if(threadIdx.x % 32 == 0) {
                start = atomicAdd(written, static_cast<unsigned long long>(size));
            }

            start = __shfl_sync(~0, start, 0);

            if(start + size > bufSize) {
                *fail = 1;
                sharedUnlockBucket(i);
                return;
            }

            K* warpKeys = keys + start;
            V* warpValues = values + start;

            unsafeReadBucket(i, warpKeys, warpValues, size);

            sharedUnlockBucket(i);
        } 
    }


    LSLAB_HOST_DEVICE unsigned size() {
        return number_of_buckets;
    }

private:
    volatile SlabData<K, V> **slabs;
    unsigned number_of_buckets;
    WarpAllocCtx<K,V> ctx;
};

}
