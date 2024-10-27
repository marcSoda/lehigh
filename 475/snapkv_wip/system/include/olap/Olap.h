#include <olap/QueryMapBase.h>
#include <olap/GPUSnapshotter.h>
#include <olap/TxSnapshotter.h>
#pragma once

namespace sikv {
namespace olap {

template<typename K, typename V, int32_t LSLAB_SIZE>
using QueryMap = QueryMapBase<K, V, GPUSnapshotter<K, V, LSLAB_SIZE>>;

}
}
