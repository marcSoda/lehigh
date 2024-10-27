/*
 * Copyright (c) 2020-2021 dePaul Miller (dsm220@lehigh.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "testheader.h"
#include <unordered_map>

using namespace lslab;

const int BLOCKS = 128;
const int THREADS_PER_BLOCK = 512;

LSLAB_HOST_DEVICE int memcmp_(const volatile void* a, const volatile void* b, size_t size) noexcept {
    for(size_t i = 0; i < size; i++) {
        char diff = reinterpret_cast<const volatile char*>(a)[i] - reinterpret_cast<const volatile char*>(b)[i];
        if(diff != 0) return diff;
    }
    return 0;
}

struct Key {

    LSLAB_HOST_DEVICE constexpr Key() : bytes{0} {}

    LSLAB_HOST_DEVICE Key(const Key& k) {
        memcpy(bytes, k.bytes, sizeof(bytes));
    }
    
    LSLAB_HOST_DEVICE Key(const volatile Key& k) {
        for(int i = 0; i < 128; i++) {
            bytes[i] = k.bytes[i];
        }
    }
    
    LSLAB_HOST_DEVICE Key(unsigned long long i) {
        memset(bytes, 0, sizeof(bytes));
        memcpy(bytes, &i, sizeof(unsigned long long));
    }

    LSLAB_HOST_DEVICE bool operator==(const Key k) const volatile {
        return memcmp_(bytes, k.bytes, 128) == 0;
    }

    LSLAB_HOST_DEVICE volatile Key& operator=(const volatile Key& k) volatile {
        for(int i = 0; i < 128; i++) {
            bytes[i] = k.bytes[i];
        }
        return *this;
    }

    LSLAB_HOST_DEVICE volatile Key& operator=(const Key& k) {
        for(int i = 0; i < 128; i++) {
            bytes[i] = k.bytes[i];
        }
        return *this;
    }

    friend std::ostream& operator<<(std::ostream&, const Key&);

    alignas(128) char bytes[128];
};

std::ostream& operator<<(std::ostream& s, const Key& k) {
    unsigned long long i = 0;
    memcpy(&i, k.bytes, sizeof(unsigned long long));
    s << i;
    return s;
}

namespace std {

template<>
struct hash<Key> {

    std::size_t operator()(const Key& k) const {
        return k.bytes[0];
    }

};

}

TEST(slabunified_test, PutRemoveTest_128B_Heavy_OtherAPI_And_Dump) {

    const int BLOCKS_ = 1;
    const int CHOSEN_THREADS_PER_BLOCK = 64;

    const int size = 1;
    std::hash<Key> hfn;
    SlabUnified<Key, int *, BLOCKS_, CHOSEN_THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<Key, int *, BLOCKS_, CHOSEN_THREADS_PER_BLOCK>();

    s.setGPU();

    std::unordered_map<Key, int*> reference;
    std::unordered_map<int*, Key> reverse;

    for (int rep = 0; rep < 10; rep++) {

        for (unsigned i = 0; i < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; i++) {
            unsigned long long key = i + 1 + rep * CHOSEN_THREADS_PER_BLOCK * BLOCKS_;
            int *value = new int[256]; // allocating 1KB
            reference[key] = value;
            reverse[value] = key;
            //std::cerr << "(" << key << "," << (void*)value << ")" << std::endl;
            for (int w = 0; w < 256; w++) {
                value[w] = rep;
            }
            b->getBatchKeys()[i] = key;
            b->getHashValues()[i] = hfn(key);
            b->getBatchRequests()[i] = REQUEST_INSERT;
            b->getBatchValues()[i] = value;
            value = nullptr;
        }
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch_traditional(b, BLOCKS_, CHOSEN_THREADS_PER_BLOCK, 0x0, rep);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        gpuErrchk(cudaPeekAtLastError());
        for (int j = 0; j < CHOSEN_THREADS_PER_BLOCK * BLOCKS_; j++) {
                delete[] b->getBatchValues()[j];
        }

        Key* allKeys;
        int** allValues;

        s.getTable(allKeys, allValues, BLOCKS, CHOSEN_THREADS_PER_BLOCK, 0x0);

    }

    delete b;
}

TEST(slabunified_test, GetAllPutRemoveTest) {

    const int size = 50;
    std::hash<unsigned> hfn;
    SlabUnified<unsigned, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) size; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < size; j++) {
                b->getBatchRequests()[j] = (rand() % 10 == 1 ? REQUEST_REMOVE : REQUEST_INSERT);
                unsigned key = j + 1;
                int *value = nullptr;
                if(b->getBatchRequests()[j] == REQUEST_INSERT) { 
                    value = new int[256]; // allocating 1KB
                    for (int w = 0; w < 256; w++) {
                        value[w] = rep;
                    }
                    std::cerr << "INSERT " << key << " " << (void*) value << std::endl;
                } else {
                    std::cerr << "REMOVE " << key << std::endl; 
                    GTEST_ASSERT_EQ(REQUEST_REMOVE, b->getBatchRequests()[j]);
                }
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchValues()[j] = value;
            }
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0, 2 * rep + 1);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != nullptr) {

                    GTEST_ASSERT_NE(b->getBatchValues()[j], nullptr);
                    for (int w = 0; w < 256; w++) {
                        GTEST_ASSERT_EQ(b->getBatchValues()[j][w], rep - 1) << " old insert was rep - 1";
                    }

                    delete[] b->getBatchValues()[j];
                } else if(b->getBatchRequests()[j] == REQUEST_REMOVE) {
                    std::cerr << "Removed " << b->getBatchKeys()[j] << " with value " << b->getBatchValues()[j] << " " << __FILE__ << ":" << __LINE__ << std::endl;
                    if(b->getBatchValues()[j] != nullptr)
                        delete[] b->getBatchValues()[j];
                }
            }
        }

        unsigned* allKeys;
        int** allValues;

        size_t size = s.getTable(allKeys, allValues, BLOCKS, THREADS_PER_BLOCK, 0x0);

        for(size_t i = 0; i < size; i++) {
            std::cerr << "GOT " << allKeys[i] << " " << (void*) allValues[i] << std::endl;
            for(int j = 0; allValues[i] && j < 256; j++) {
                GTEST_ASSERT_EQ(allValues[i][j], rep) << " last insert was rep";
            }
        }

        delete[] allKeys;
        delete[] allValues;
    }

    delete b;
}

TEST(slabunified_test, GetAllPutRemoveTestHeavy) {

    const int size = 2;
    const int toInsert = 50;
    std::hash<unsigned> hfn;
    SlabUnified<unsigned, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) toInsert; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < toInsert; j++) {
                b->getBatchRequests()[j] = (rand() % 10 == 1 ? REQUEST_REMOVE : REQUEST_INSERT);
                unsigned key = j + 1;
                int *value = nullptr;
                if(b->getBatchRequests()[j] == REQUEST_INSERT) { 
                    value = new int[256]; // allocating 1KB
                    for (int w = 0; w < 256; w++) {
                        value[w] = rep;
                    }
                    std::cerr << "INSERT " << key << " " << (void*) value << std::endl;
                } else {
                    std::cerr << "REMOVE " << key << std::endl; 
                    GTEST_ASSERT_EQ(REQUEST_REMOVE, b->getBatchRequests()[j]);
                }
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchValues()[j] = value;
            }
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0, 2 * rep);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != nullptr) {

                    GTEST_ASSERT_NE(b->getBatchValues()[j], nullptr);
                    for (int w = 0; w < 256; w++) {
                        GTEST_ASSERT_EQ(b->getBatchValues()[j][w], rep - 1) << " old insert was rep - 1";
                    }

                    delete[] b->getBatchValues()[j];
                } else if(b->getBatchRequests()[j] == REQUEST_REMOVE) {
                    std::cerr << "Removed " << b->getBatchKeys()[j] << " with value " << b->getBatchValues()[j] << " " << __FILE__ << ":" << __LINE__ << std::endl;
                    delete[] b->getBatchValues()[j];
                }
            }
        }

        unsigned* allKeys;
        int** allValues;

        size_t size = s.getTable(allKeys, allValues, BLOCKS, THREADS_PER_BLOCK, 0x0);

        for(size_t i = 0; i < size; i++) {
            std::cerr << "GOT " << allKeys[i] << " " << (void*) allValues[i] << std::endl;
            if(allValues[i]) {
                for(int j = 0; j < 256; j++) {
                    GTEST_ASSERT_EQ(allValues[i][j], rep) << " last insert was rep";
                }
            }
        }

        delete[] allKeys;
        delete[] allValues;
    }

    delete b;
}

TEST(slabunified_test, DISABLED_GetAllPutRemoveTestHeavy128B) {

    const int size = 2;
    const int toInsert = 50;
    std::hash<Key> hfn;
    SlabUnified<Key, int *, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<Key, int *, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();

    for (int rep = 0; rep < 100; rep++) {

        for (unsigned i = 0; i < (unsigned) toInsert; i += THREADS_PER_BLOCK * BLOCKS) {
            unsigned j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS && i * THREADS_PER_BLOCK * BLOCKS + j < toInsert; j++) {
                b->getBatchRequests()[j] = (rand() % 10 == 1 ? REQUEST_REMOVE : REQUEST_INSERT);
                unsigned key = j + 1;
                int *value = nullptr;
                if(b->getBatchRequests()[j] == REQUEST_INSERT) { 
                    value = new int[256]; // allocating 1KB
                    for (int w = 0; w < 256; w++) {
                        value[w] = rep;
                    }
                    std::cerr << "INSERT " << key << " " << (void*) value << std::endl;
                } else {
                    std::cerr << "REMOVE " << key << std::endl; 
                    GTEST_ASSERT_EQ(REQUEST_REMOVE, b->getBatchRequests()[j]);
                }
                b->getBatchKeys()[j] = key;
                b->getHashValues()[j] = hfn(key);
                b->getBatchValues()[j] = value;
            }
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                b->getBatchRequests()[j] = REQUEST_EMPTY;
            }
            s.moveBufferToGPU(b, 0x0);
            s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0, 2 * rep);
            s.moveBufferToCPU(b, 0x0);
            gpuErrchk(cudaStreamSynchronize(0x0));
            j = 0;
            for (; j < THREADS_PER_BLOCK * BLOCKS; j++) {
                if (b->getBatchRequests()[j] == REQUEST_INSERT && b->getBatchValues()[j] != nullptr) {

                    GTEST_ASSERT_NE(b->getBatchValues()[j], nullptr);
                    for (int w = 0; w < 256; w++) {
                        GTEST_ASSERT_EQ(b->getBatchValues()[j][w], rep - 1) << " old insert was rep - 1 for key " << j + 1 << " pointer returned was " << reinterpret_cast<void*>(b->getBatchValues()[j]);
                    }

                    delete[] b->getBatchValues()[j];
                } else if(b->getBatchRequests()[j] == REQUEST_REMOVE) {
                    std::cerr << "Removed " << b->getBatchKeys()[j] << " with value " << b->getBatchValues()[j] << " " << __FILE__ << ":" << __LINE__ << std::endl;
                    delete[] b->getBatchValues()[j];
                }
            }
        }

        Key* allKeys;
        int** allValues;

        size_t size = s.getTable(allKeys, allValues, BLOCKS, THREADS_PER_BLOCK, 0x0);

        for(size_t i = 0; i < size; i++) {
            std::cerr << "GOT " << allKeys[i] << " " << (void*) allValues[i] << std::endl;
            if(allValues[i] != nullptr) {
                for(int j = 0; j < 256; j++) {
                    GTEST_ASSERT_EQ(allValues[i][j], rep) << " last insert was rep";
                }
            }
            std::cerr << "Next" << std::endl;
        }

        delete[] allKeys;
        delete[] allValues;
    }

    delete b;
}

