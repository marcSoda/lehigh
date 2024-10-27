#include "LSlab.h"
#include <cstdio>
#include <culog/culog.h>
#include "shfl.h"
#include <GroupAllocator/groupallocator>
#include <type_traits>

#pragma once

namespace lslab {

enum Operation {
    NOP = 0,
    GET = 2,
    PUT = 1,
    REMOVE = 3
};

#define SEARCH_NOT_FOUND 0
#define ADDRESS_LANE 32
#define VALID_KEY_MASK 0x7fffffffu
#define DELETED_KEY 0

const unsigned long long EMPTY_POINTER = 0;
#define BASE_SLAB 0

template<typename K, typename V>
struct SlabData {

    using KSub = typename std::conditional<sizeof(K) < sizeof(unsigned long long), unsigned long long, K>::type;

    static_assert(alignof(KSub) % alignof(void*) == 0, "Alignment must be a multiple of the pointer alignment");

    union {
        int ilock = 0;
        alignas(128) char p[128];
    }; // 128 bytes

    alignas(128) KSub key[32] = {K{}}; // 256 byte

    alignas(128) V value[32] = {V{}}; // 256 byte
    // the 32nd element is next

    alignas(128) uint64_t version[32] = {};
    
    uint32_t tombstone = 0;

    /**
     * There is a barrier after this locking
     */
    LSLAB_DEVICE void LockSlab() volatile {
        if (threadIdx.x % 32 == 0) {
            while (atomicCAS(const_cast<int*>(&ilock), 0, -1) != 0);
        }
        __syncwarp();
    }

    /**
     * There is a barrier after this locking
     */
    LSLAB_DEVICE void SharedLockSlab() volatile {
    
        if (threadIdx.x % 32 == 0) {
            while (true) {
                auto pred = ilock;
    
                if (pred != -1 && atomicCAS(const_cast<int*>(&ilock), pred, pred + 1) == pred) {
                    break;
                }
            }
        }
        __syncwarp();
    }
    
    /**
     * Note there is no barrier before or after, pay attention to reordering
     */
    LSLAB_DEVICE void UnlockSlab() volatile {
        __threadfence();
        if (threadIdx.x % 32 == 0) {
            atomicExch(const_cast<int*>(&ilock), 0);
        }
    }
    
    /**
     * Note there is no barrier before or after, pay attention to reordering
     */
    LSLAB_DEVICE void SharedUnlockSlab() volatile {
        __threadfence();
        if (threadIdx.x % 32 == 0) {
            atomicAdd(const_cast<int*>(&ilock), -1);
        }
    }
    
    LSLAB_DEVICE volatile KSub* AddressKey(const unsigned laneId = threadIdx.x % 32) volatile {
        static_assert(sizeof(KSub) >= sizeof(void*), "Need to be able to substitute pointers for values");
        return &key[laneId];
    }

    LSLAB_DEVICE volatile uint64_t* AddressVersion(const unsigned laneId = threadIdx.x % 32) volatile {
        return &version[laneId];
    }
 
    LSLAB_DEVICE uint64_t ReadVersion(const unsigned laneId = threadIdx.x % 32) volatile {
        return version[laneId];
    }
    
    LSLAB_DEVICE bool IsDeleted(const unsigned laneId = threadIdx.x % 32) volatile {
        return (tombstone >> laneId) & 0x1;
    }
    
    LSLAB_DEVICE void ChangeMarkDeleted(const unsigned laneId = threadIdx.x % 32) volatile {
        tombstone ^= (0x1u << laneId);
    }
 
    LSLAB_DEVICE void MarkDeleted(const unsigned laneId = threadIdx.x % 32) volatile {
        tombstone |= (0x1u << laneId);
    }
  
    LSLAB_DEVICE void MarkNotDeleted(const unsigned laneId = threadIdx.x % 32) volatile {
        tombstone &= ~(0x1u << laneId);
    }  

    LSLAB_DEVICE volatile V* AddressValue(const unsigned laneId = threadIdx.x % 32) volatile {
        return &value[laneId];
    }

    LSLAB_DEVICE volatile SlabData<K, V>* ReadNext() volatile {
        return *AddressNext();
    }

    LSLAB_DEVICE volatile SlabData<K, V>* volatile* AddressNext() volatile {
        auto keyPtr = AddressKey(ADDRESS_LANE - 1);
        return reinterpret_cast<volatile SlabData<K, V>* volatile*>(keyPtr);
    }
};

template<typename K, typename V>
struct SlabCtx {

    SlabCtx(groupallocator::GroupAllocator &gAlloc, unsigned size, int gpuid, cudaStream_t stream) {
        gpuErrchk(cudaSetDevice(gpuid));
        num_of_buckets = size;
        std::cerr << "Size of index is " << size << std::endl;
        std::cerr << "Each slab is " << sizeof(SlabData<K, V>) << "B" << std::endl;

        gAlloc.allocate(&slabs, sizeof(void *) * num_of_buckets, false);

        for (int i = 0; i < num_of_buckets; i++) {
            gAlloc.allocate(&slabs[i], sizeof(SlabData<K, V>), false);

            static_assert(sizeof(slabs[i]->key[0]) >= sizeof(void *),
                          "The key size needs to be greater or equal to the size of a memory address");

            memset((void *) (slabs[i]), 0, sizeof(SlabData<K, V>));

            for (int j = 0; j < 31; j++) {
                const_cast<typename SlabData<K, V>::KSub *>(slabs[i]->key)[j] = K{};
            }

            volatile K* ptrToAddr = reinterpret_cast<volatile K*>(slabs[i]->key + 31);

            *reinterpret_cast<void**>(const_cast<K*>(ptrToAddr)) = nullptr;

            for (int j = 0; j < 32; j++) {
                const_cast<V&>(slabs[i]->value[j]) = V{};
                const_cast<uint64_t&>(slabs[i]->version[j]) = 0;
            }

        }

        gAlloc.moveToDevice(gpuid, stream);

        gpuErrchk(cudaDeviceSynchronize())

        std::cerr << "Size allocated for Slab: "
                  << gAlloc.pagesAllocated() * 4.0 / 1024.0 / 1024.0 << "GB"
                  << std::endl;

    }

    volatile SlabData<K, V> **slabs;
    unsigned num_of_buckets;
};

}
