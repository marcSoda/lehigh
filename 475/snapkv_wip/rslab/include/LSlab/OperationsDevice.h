#include "LSlab.h"
#include <cstdio>
#include <culog/culog.h>
#include "Pool.h"
#include <cassert>

#pragma once

namespace lslab {

template<typename K, typename V>
LSLAB_DEVICE void warp_operation_unsafe_search(bool &is_active, const K &myKey,
                                                      V &myValue, const unsigned &modhash,
                                                      volatile SlabData<K, V> **__restrict__ slabs,
                                                      unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    volatile SlabData<K, V>* next = nullptr;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    const K threadKey = myKey;
    unsigned last_work_queue = 0;

    unsigned src_lane;
    K src_key;
    unsigned src_bucket;

    while (work_queue != 0) {

        if(work_queue != last_work_queue) {
            src_lane = __ffs(work_queue) - 1;
            src_key = shfl(~0u, threadKey, src_lane);
            src_bucket = shfl(~0u, modhash, src_lane);
            next = slabs[src_bucket];
        }
        
        auto read_key = *reinterpret_cast<volatile K*>(next->AddressKey());

        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;

        if (masked_ballot != 0) {
            auto read_value = *next->AddressValue();

            unsigned found_lane = __ffs(masked_ballot) - 1;
            auto found_value = shfl(~0u, read_value, found_lane);
            if (laneId == src_lane) {
                myValue = found_value;
                is_active = false;
            }
        } else {
            auto next_ptr = next->ReadNext();
            if (next_ptr == 0) {
                if (laneId == src_lane) {
                    myValue = SEARCH_NOT_FOUND;
                    is_active = false;
                }
            } else {
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
    }
}


template<typename K, typename V>
LSLAB_DEVICE void warp_operation_search(bool &is_active, const K &myKey,
                                                      V &myValue, const unsigned &modhash,
                                                      volatile SlabData<K, V> **__restrict__ slabs,
                                                      unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    volatile SlabData<K, V>* next = nullptr;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    const K threadKey = myKey;
    unsigned last_work_queue = 0;

    unsigned src_lane;
    K src_key;
    unsigned src_bucket;

    while (work_queue != 0) {

        if(work_queue != last_work_queue) {
            src_lane = __ffs(work_queue) - 1;
            src_key = shfl(~0u, threadKey, src_lane);
            src_bucket = shfl(~0u, modhash, src_lane);
            next = slabs[src_bucket];
            next->SharedLockSlab();
            __threadfence();
        }
        
        auto read_key = *reinterpret_cast<volatile K*>(next->AddressKey());

        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;

        if (masked_ballot != 0) {
            auto read_value = *next->AddressValue();

            unsigned found_lane = __ffs(masked_ballot) - 1;
            auto found_value = shfl(~0u, read_value, found_lane);
            if (laneId == src_lane) {
                myValue = found_value;
                is_active = false;
            }
        } else {
            auto next_ptr = next->ReadNext();
            if (next_ptr == 0) {
                if (laneId == src_lane) {
                    myValue = SEARCH_NOT_FOUND;
                    is_active = false;
                }
            } else {
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);

        if(work_queue != last_work_queue){
            slabs[src_bucket]->SharedUnlockSlab();
        }
    }
}

/**
 * Returns value when removed or empty on removal
 * @tparam K
 * @tparam V
 * @param is_active
 * @param myKey
 * @param myValue
 * @param modhash
 * @param slabs
 * @param num_of_buckets
 */
template<typename K, typename V>
LSLAB_DEVICE void
warp_operation_delete(bool &is_active, const K &myKey,
                      V &myValue, const unsigned &modhash, const uint64_t& myVersion,
                      volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets) {
    const unsigned laneId = threadIdx.x & 0x1Fu;
    volatile SlabData<K, V>* next = nullptr;
    unsigned work_queue = __ballot_sync(~0u, is_active);
    unsigned last_work_queue = 0;

    unsigned src_lane;
    K src_key;
    unsigned src_bucket;


    while (work_queue != 0) {
        
        if(work_queue != last_work_queue){
            src_lane = __ffs(work_queue) - 1;
            src_key = shfl(~0u, myKey, src_lane);
            src_bucket = shfl(~0u, modhash, src_lane);
            next = slabs[src_bucket];
            next->LockSlab();
            __threadfence();
        }

        auto read_key = *reinterpret_cast<volatile K*>(next->AddressKey());

        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;

        if (masked_ballot != 0) {
            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;
                if(next->ReadVersion(dest_lane) < myVersion) {
                    // TODO reconcile old method
                    //K* ptr = const_cast<K*>(reinterpret_cast<volatile K*>(next->AddressKey(dest_lane)));
                    //*ptr = K{};
                    auto vptr = next->AddressValue(dest_lane);
                    myValue = *vptr;
                    *vptr = V{};
                    *(next->AddressVersion(dest_lane)) = myVersion;
                    next->MarkDeleted(dest_lane);
                } 
                is_active = false;
                __threadfence();
            }
        } else {
            auto next_ptr = next->ReadNext();
            if (next_ptr == nullptr) {
                if(src_lane == laneId) {
                    is_active = false;
                    myValue = V{};
                }
            } else {
                __syncwarp();
                next = next_ptr;
            }
        }

        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);
        if(work_queue != last_work_queue){
            slabs[src_bucket]->UnlockSlab();
        }
    }
}

template<typename K, typename V>
LSLAB_DEVICE void
warp_operation_replace(bool &is_active, const K &myKey,
                       V &myValue, const unsigned &modhash, const uint64_t &myVersion,
                       volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx<K, V> ctx) {
    
    using KSub = typename SlabData<K, V>::KSub;
    
    unsigned src_lane;
    K src_key;
    unsigned src_bucket;

    const unsigned laneId = threadIdx.x & 0x1Fu;
    volatile SlabData<K, V>* next = nullptr;
    unsigned work_queue = __ballot_sync(~0u, is_active);

    unsigned last_work_queue = 0;

    volatile SlabData<K, V>* empty_next = nullptr;

    while (work_queue != 0) {

        if(work_queue != last_work_queue) {
            src_lane = __ffs(work_queue) - 1;
            src_key = shfl(~0u, myKey, src_lane);
            src_bucket = shfl(~0u, modhash, src_lane);

            empty_next = nullptr;
            next = slabs[src_bucket];
            assert(next != nullptr);
            next->LockSlab();
            CULOGF(laneId == 0, "(%u,%u) locked %u line %d\n", threadIdx.x, blockIdx.x, src_bucket, __LINE__);
        }

        CULOGF(laneId == 0, "Pointer: %p thread (%u,%u) bucket %u line %d\n", next, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, __LINE__);
        assert(next != nullptr);
        __threadfence();
        auto keyPtr = next->AddressKey();
        auto read_key = *reinterpret_cast<volatile K*>(keyPtr);
        bool to_share = read_key == src_key;
        auto masked_ballot = __ballot_sync(~0u, to_share) & VALID_KEY_MASK;

        if(empty_next == nullptr && read_key == K{}){
            empty_next = next;
        }

        if (masked_ballot != 0) {
            if (src_lane == laneId) {
                unsigned dest_lane = __ffs(masked_ballot) - 1;
                assert(dest_lane != ADDRESS_LANE - 1);
                // TODO reconcile with older version
                if(next->ReadVersion(dest_lane) < myVersion) {
                    auto *addrValue = next->AddressValue(dest_lane);
                    V tmpValue = *(addrValue);
                    *(addrValue) = myValue;
                    myValue = tmpValue;
                    *(next->AddressVersion(dest_lane)) = myVersion;
                    next->MarkNotDeleted(dest_lane);
                }
                is_active = false;
                __threadfence();
            }
        } else {
            volatile SlabData<K, V>* next_ptr = next->ReadNext();
            CULOGF(laneId == 0, "Next Pointer: %p thread (%u,%u) bucket %u line %d\n", next_ptr, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, __LINE__);
            if (next_ptr == nullptr) {
                __threadfence_system();
                masked_ballot = (int) (__ballot_sync(~0u, empty_next != nullptr) & VALID_KEY_MASK);
                if (masked_ballot != 0) {
                    unsigned dest_lane = __ffs(masked_ballot) - 1;
                    assert(dest_lane != ADDRESS_LANE - 1);
                    volatile SlabData<K, V>* new_empty_next = shfl(~0, empty_next, dest_lane);
                    CULOGF(laneId == 0, "Thread (%u,%u) bucket %u inserting into slab[%u] at %p after next_ptr was null line %d\n", static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, dest_lane, new_empty_next,  __LINE__);
                    if (src_lane == laneId) {
                        auto addrKey = new_empty_next->AddressKey(dest_lane);
                        volatile V *addrValue = new_empty_next->AddressValue(dest_lane);
                        CULOGF(true, "Thread (%u,%u) bucket %u updating %p line %d\n", static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, reinterpret_cast<volatile void*>(addrKey),  __LINE__);
                        *(reinterpret_cast<volatile K*>(addrKey)) = src_key;
                        *(addrValue) = myValue;
                        *(new_empty_next->AddressVersion(dest_lane)) = myVersion;
                        new_empty_next->MarkNotDeleted(dest_lane);
                        myValue = V{};
                        is_active = false;
                        __threadfence_system();
                    }
                } else {
                    volatile SlabData<K, V>* new_slab_ptr = ctx.allocate();
                    assert(new_slab_ptr != nullptr);
                    if (laneId == 0) {
                        auto slabAddr = next->AddressNext();
                        *slabAddr = new_slab_ptr;
                        CULOGF(true, "Thread (%u, %u) wrote %p line %d\n", threadIdx.x, blockIdx.x, new_slab_ptr, __LINE__);
                        __threadfence_system();
                    }
                    CULOGF(laneId == 0, "Thread (%u, %u) bucket %u allocating %p after next_ptr was null line %d\n", static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, new_slab_ptr, __LINE__);
                    __syncwarp();
                    next = new_slab_ptr;
                    assert(next != nullptr);
                }
            } else {
                next = next_ptr;
                assert(next != nullptr);
            }
        }
        last_work_queue = work_queue;

        work_queue = __ballot_sync(~0u, is_active);

        if (work_queue != last_work_queue) {
            slabs[src_bucket]->UnlockSlab();
            CULOGF(laneId == 0, "(%u,%u) unlocked %u line %d\n", threadIdx.x, blockIdx.x, src_bucket, __LINE__);
        }
    }
}

/**
 * Returns value when removed or empty on removal
 * @tparam K
 * @tparam V
 * @param is_active
 * @param myKey
 * @param myValue
 * @param modhash
 * @param slabs
 * @param num_of_buckets
 */
//template<typename K, typename V>
//LSLAB_DEVICE void
//warp_operation_delete_or_replace(bool &is_active, const K &myKey,
//                      V &myValue, const unsigned &modhash,
//                      volatile SlabData<K, V> **__restrict__ slabs, unsigned num_of_buckets, WarpAllocCtx<K, V> ctx, Operation op) {
//
//    using KSub = typename SlabData<K, V>::KSub;
//    
//    const unsigned laneId = threadIdx.x & 0x1Fu;
//    unsigned long long next = BASE_SLAB;
//    unsigned work_queue = __ballot_sync(~0u, is_active);
//    unsigned last_work_queue = 0;
//    bool foundEmptyNext = false;
//    unsigned long long empty_next = BASE_SLAB;
//
//    while (work_queue != 0) {
//        next = (work_queue != last_work_queue) ? (BASE_SLAB) : next;
//        auto src_lane = __ffs(work_queue) - 1;
//        auto src_key = shfl(~0u, myKey, src_lane);
//        unsigned src_bucket = shfl(~0u, modhash, (int) src_lane);
//        
//        if(work_queue != last_work_queue){
//            foundEmptyNext = false;
//            empty_next = 0x0;
//            LockSlab(BASE_SLAB, src_bucket, laneId, slabs);
//        }
//
//        CULOGF(laneId == 0, "Pointer: %p thread (%u, %u) bucket %u\n", next, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket);
//        KSub read_key = ReadSlabKey(next, src_bucket, laneId, slabs);
//
//        auto masked_ballot = __ballot_sync(~0u, read_key == src_key) & VALID_KEY_MASK;
//        
//        if(!foundEmptyNext && read_key == K{}){
//            foundEmptyNext = true;
//            CULOGF(true, "Found empty next ptr: %p thread (%u, %u) bucket %u\n", next, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket);
//            empty_next = next;
//        }
//        
//        if (masked_ballot != 0) {
//
//            if (src_lane == laneId) {
//                unsigned dest_lane = __ffs(masked_ballot) - 1;
//
//                if(op == REMOVE) {
//                    *const_cast<KSub *>(SlabAddressKey(next, src_bucket, dest_lane, slabs)) = K{};
//                    is_active = false;
//                    myValue = ReadSlabValue(next, src_bucket, dest_lane, slabs);
//                    //success = true;
//                } else {
//                    volatile KSub *addrKey = SlabAddressKey(next, src_bucket, dest_lane, slabs);
//                    auto *addrValue = SlabAddressValue(next, src_bucket, dest_lane, slabs);
//                    V tmpValue = V{};
//                    K addrKeyDeref = const_cast<KSub&>(*addrKey);
//                    if (addrKeyDeref == K{}) {
//                        *const_cast<KSub*>(addrKey) = myKey;
//                    } else {
//                        tmpValue = *const_cast<V*>(addrValue);
//                    }
//                    *const_cast<V*>(addrValue) = myValue;
//                    myValue = tmpValue;
//                    is_active = false;
//                }
//                __threadfence_system();
//            }
//
//        } else {
//            static_assert(sizeof(read_key) >= sizeof(void*), "Need read key to be bigger than the size of a pointer");
//            unsigned long long next_ptr = ReadNext(next, src_bucket, laneId, slabs);
//            CULOGF(laneId == 0, "Next_pointer: %p thread (%u, %u) bucket %u\n", next, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket);
//            if (next_ptr == 0) {
//                if(op == REMOVE) {
//                    is_active = false;
//                    myValue = V{};
//                } else {
//                    __threadfence_system();
//                    masked_ballot = (int) (__ballot_sync(~0u, foundEmptyNext) & VALID_KEY_MASK);
//                    if (masked_ballot != 0) {
//                        unsigned dest_lane = __ffs(masked_ballot) - 1;
//                        CULOGF(true, "Empty_next: %p thread (%u, %u) bucket %u from %u\n", empty_next, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, dest_lane);
//                        unsigned long long new_empty_next = shfl(~0u, empty_next, dest_lane);
//                        CULOGF(true, "New_empty_pointer: %p thread (%u, %u) bucket %u from %u\n", new_empty_next, static_cast<unsigned>(threadIdx.x), static_cast<unsigned>(blockIdx.x), src_bucket, dest_lane);
//                        if (src_lane == laneId) {
//                            auto addrKey = SlabAddressKey(new_empty_next, src_bucket, dest_lane, slabs);
//                            auto addrValue = SlabAddressValue(new_empty_next, src_bucket, dest_lane, slabs);
//                            V tmpValue = V{};
//                            K addrKeyDeref = const_cast<KSub&>(*addrKey);
//                            if (addrKeyDeref == K{}) {
//                                *const_cast<KSub*>(addrKey) = src_key;
//                            } else {
//                                tmpValue = *const_cast<V*>(addrValue);
//                            }
//                            *const_cast<V*>(addrValue) = myValue;
//                            myValue = tmpValue;
//                            __threadfence_system();
//                            is_active = false;
//                        }
//                    } else {
//                        unsigned long long new_slab_ptr = ctx.allocate();
//                        if (laneId == ADDRESS_LANE - 1) {
//                            auto *slabAddr = SlabAddressKey(next, src_bucket, ADDRESS_LANE - 1, slabs);
//                            *reinterpret_cast<volatile unsigned long long*>(slabAddr) = new_slab_ptr;
//                            __threadfence_system();
//                        }
//                        next = new_slab_ptr;
//                    }
//                }
//                //success = false;
//            } else {
//                next = next_ptr;
//            }
//        }
//
//        last_work_queue = work_queue;
//
//        work_queue = __ballot_sync(~0u, is_active);
//        if(work_queue != last_work_queue){
//            UnlockSlab(BASE_SLAB, src_bucket, laneId, slabs);
//        }
//    }
//}

}
