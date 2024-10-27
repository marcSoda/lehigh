#include <olap/QueryMapBase.h>

#pragma once

namespace sikv {
namespace olap {

template<typename K, typename V, typename SnapshotterT>
class QueryPolicy {
public: 
    std::pair<void*, size_t> query(std::string name, const std::vector<K>& keys, const std::vector<V>& values, void* args) {
        thread_local cudaStream_t stream = 0x0;
        if(stream == 0) {
            gpuErrchk(cudaStreamCreate(&stream));
        }

        K* k_keys;
        V* k_values;
        gpuErrchk(cudaMallocAsync(&k_keys, sizeof(K) * keys.size(), stream));
        gpuErrchk(cudaMemcpyAsync(k_keys, keys.data(), sizeof(K) * keys.size(), cudaMemcpyHostToDevice, stream));
        gpuErrchk(cudaMallocAsync(&k_values, sizeof(V) * keys.size(), stream));
        gpuErrchk(cudaMemcpyAsync(k_values, values.data(), sizeof(V) * keys.size(), cudaMemcpyHostToDevice, stream));

        size_t size = keys.size();
        Snapshot<K, V> snapshot{k_keys, k_values, size, nullptr, nullptr};
        unsigned long long nranges = 1;
        int ksize = keys.size();
        return qmap->template runKernels<false>(snapshot, size, nranges, ksize, args, name, stream);
    }
protected:
    QueryPolicy(QueryMapBase<K, V, SnapshotterT>* qmap_) : qmap(qmap_) {
        
    }

    ~QueryPolicy() = default;

private:
    QueryMapBase<K, V, SnapshotterT>* qmap;
};

}
}
