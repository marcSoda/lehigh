#include <VariadicType.h>
#include <SiKV.h>
#include <device/Kernel.h>
#include <device/GPUMap.h>
#include <vector>
#include <unordered_map>
#include <dlfcn.h>
#include <link.h>
#include <string>
#include <cuda.h>
#include <fstream>
#include <sstream>
#include <elf.h>
#include <cstring>
#include <oltp/TransactionalMap.h>

#pragma once

namespace sikv {
namespace olap {

template<typename K, typename V>
struct Snapshot {

    Snapshot(K* keys_, V* values_, unsigned long long size_, unsigned long long* rangeStart_, unsigned long long* rangeEnd_) : keys(keys_), values(values_), size(size_), rangeStart(rangeStart_), rangeEnd(rangeEnd_) {}
   
    Snapshot(const Snapshot<K, V>&) = delete;
    Snapshot<K, V>& operator=(const Snapshot<K, V>&) = delete;

    Snapshot(Snapshot<K, V>&& other) {
        keys = other.keys;
        values = other.values;
        rangeStart = other.rangeStart;
        rangeEnd = other.rangeEnd;
        other.keys = nullptr;
        other.values = nullptr;
        other.rangeStart = nullptr;
        other.rangeEnd = nullptr;
    }

    Snapshot<K,V>& operator=(Snapshot<K, V>&& other) {
        if(this == &other) return *this;
        keys = other.keys;
        values = other.values;
        rangeStart = other.rangeStart;
        rangeEnd = other.rangeEnd;
        other.keys = nullptr;
        other.values = nullptr;
        other.rangeStart = nullptr;
        other.rangeEnd = nullptr;
    }
    
    ~Snapshot() {
        if(keys)
            gpuErrchk(cudaFree(keys));
        if(values)
            gpuErrchk(cudaFree(values));
        if(rangeStart)
            gpuErrchk(cudaFree(rangeStart));
        if(rangeEnd)
            gpuErrchk(cudaFree(rangeEnd));
    }
    
    K* keys;
    V* values;
    unsigned long long size;
    unsigned long long* rangeStart;
    unsigned long long* rangeEnd;
};
}
}
