/**
 * @file
 */
#include <optional.h>
#include <SiKV.h>

#pragma once

namespace sikv {

namespace consumer {

template<typename K, typename V>
struct NoopLogConsumer {

    SIKV_HOST V** allocateValues(size_t size) {
        return new V*[size];
    }

    SIKV_HOST K* allocateKeys(size_t size) {
        return new K[size];
    }

    SIKV_HOST uint64_t* allocateVersions(size_t) { return nullptr; }

    SIKV_HOST void copyKeys(K* gpuPtr, const K* cpuPointer, size_t elements) {
        memcpy(gpuPtr, cpuPointer, sizeof(K) * elements);
    }

    SIKV_HOST void copyValues(V** gpuPtr, const V* const* cpuPointer, size_t elements) {
        memcpy(gpuPtr, cpuPointer, sizeof(V*) * elements);
    }

    SIKV_HOST void copyVersions(uint64_t*, const uint64_t*, size_t) {}

    SIKV_HOST void operator()(K*, V**, uint64_t*, size_t) {}

    inline V* constructAndAlloc(const V&) {
        return nullptr;
    }

    inline void destruct(V*) {
        return;
    }


};
}

}
