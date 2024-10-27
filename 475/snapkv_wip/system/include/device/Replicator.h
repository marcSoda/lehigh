#include <SiKV.h>
#include <cstdint>
#include <LSlab/LSlabMap.h>
#include <cooperative_groups.h>
#include <optional.h>
#include <culog/culog.h>

#pragma once

namespace sikv {

namespace device {

template<typename K>
SIKV_DEVICE K loadCS(K* k) {
//#if defined(__NVCC__) || defined(__CUDACC_RTC__)
//    return __ldcs(k);
//#else
    #pragma error "Optimizations not neccessarily enabled"
    return *k;
//#endif
}

template<>
SIKV_DEVICE uint64_t loadCS<uint64_t>(uint64_t* k) {
    uint64_t retVal;
    asm volatile("ld.global.cs.u64 %0, [%1];": "=l"(retVal): "l"(k));
    return retVal;
}

template<typename K, typename V, int32_t LSLAB_SIZE>
struct Replicator {

    using Map = lslab::LSlab<K, V*>;

    static_assert(LSLAB_SIZE % 2 == 0, "LSlab size need to be divisible by 2");

    SIKV_DEVICE Replicator(int32_t ops_, K* keys_, V** values_, uint64_t* versions_, Map lslab_) : ops(ops_), keys(keys_), values(values_), versions(versions_), lslab(lslab_) {
        assert(LSLAB_SIZE == lslab.size());
    }
    
    SIKV_DEVICE ~Replicator() {}

    SIKV_DEVICE void run() {
        //auto g = cooperative_groups::this_grid();
        //assert(g.is_valid());
        
        const int32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
        size_t h;
        const int threads = gridDim.x * blockDim.x;
        K key{};
        V* value{};

        for(int i = tid; i < ops + (32 - ops % 32); i+= threads) {
            bool mask = i < ops;
            key = mask ? loadCS<K>(keys + i) : K{};
            value = mask ? values[i] : nullptr;
            h = sikv::orderHashToBucket<K, LSLAB_SIZE>(key);
            bool mask2 = mask && value != nullptr;
            uint64_t version = mask ? versions[i] : 0;
            //if(mask2)
            //    printf("Bucket %llu\n", h);
            CULOGF(mask2, "Putting %llu %p with version %d\n", key, value, version);
            lslab.put(key, value, h, version, mask2);
            mask2 = mask && value == nullptr;
            lslab.remove(key, value, h, version, mask2);
            if(mask) {
                values[i] = value;
            }
        }
    }

    int32_t ops;
    K* keys;
    V** values;
    uint64_t* versions;
    Map lslab;
};

}
}
