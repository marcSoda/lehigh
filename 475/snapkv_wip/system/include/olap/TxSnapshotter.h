#include <olap/Snapshot.h>
#include <consumer/NoopLogConsumer.h>

#pragma once

namespace sikv {
namespace olap {

template<typename K, typename V, typename TxMap>
struct TxSnapshotter {
    using RangeArg_t = typename OrderPreservingHashBase<K>::RangeArg_t;

    TxSnapshotter(TxMap* map) : m(map) {

    }

    ~TxSnapshotter() {}

    Snapshot<K, V> takeSnapshot(const std::vector<K>& keys) {
        void* tx = m->startTx();

        std::vector<V> values;
        values.reserve(keys.size());

        for(auto& k : keys) {
            auto v = m->read(k, tx);
            if(v == cuda::std::nullopt) {
                values.push_back(V{});
            } else {
                values.push_back(*v);
            }
        }

        m->readOnlyCommit(tx);


        V* k_values;
        K* k_keys;
        unsigned long long size = keys.size();
        gpuErrchk(cudaMalloc(&k_keys, sizeof(K) * size));
        gpuErrchk(cudaMemcpy(k_keys, keys.data(), sizeof(K) * size, cudaMemcpyHostToDevice));
        gpuErrchk(cudaMalloc(&k_values, sizeof(V) * size));
        gpuErrchk(cudaMemcpy(k_values, values.data(), sizeof(V) * size, cudaMemcpyHostToDevice));
        return Snapshot<K, V>{k_keys, k_values, size, nullptr, nullptr};
    }

    TxMap* m;
};
}
}
