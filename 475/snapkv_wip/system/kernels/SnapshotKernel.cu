#include <SiKV.h>
#include <LSlab/LSlabMap.h>
#include <device/Replicator.h>
#include <optional.h>
#include <QueryStub.h>

template<typename K, typename V, int32_t LSLAB_SIZE>
__global__ void snapshotKernel(int32_t size, K* __restrict__ keys, V* __restrict__ values, ::lslab::LSlab<K,V*> lslab) {
    int32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool mask = (idx >= size);
    K key = (!mask) ? keys[idx] : K{};
    V* value{};
    int32_t h = sikv::hash<K>{}(key) % LSLAB_SIZE;
    lslab.get(key, value, h, mask);
    values[idx] = (!mask && value) ? *value : V{};
}

template<>
__global__ void snapshotKernel<unsigned long long, unsigned long long, 1024>(int32_t size, unsigned long long* __restrict__ keys, unsigned long long* __restrict__ values, ::lslab::LSlab<unsigned long long,unsigned long long*> lslab);

extern "C" __global__ void test1() {
    printf("Works");
};

static CUfunction func;

extern "C" void setup(CUmodule hmod) {
    drvErrchk(cuModuleGetFunction(&func, hmod, "test1"));
}

extern "C" sikv::ComputeBuf test(void*, void*, void*, CUstream stream) {

    void* args[] = {};

    drvErrchk(cuLaunchKernel(func, 1, 1, 1, 1, 1, 1, 0, stream, args, nullptr));

    drvErrchk(cuCtxSynchronize());

    return {nullptr, 0};
} 
