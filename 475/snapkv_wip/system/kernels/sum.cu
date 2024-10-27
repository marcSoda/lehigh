#include <QueryKernel.h>
#include <cstdint>
#include <sum.h>

Kernel k[] = {
    {KernelType::GPU, MemType::DEVICE, MemType::DEVICE, ""}
};

extern "C" void getKernels(Kernel** kernels, int* size) {
    *size = 1;
    *kernels = k;
}

extern "C" __global__ void computeKernel(int size, void*, void* value_, void*, void* output, unsigned long long*, unsigned long long*, unsigned long long) {
    
    uint64_t* value = reinterpret_cast<uint64_t*>(value_);

    __shared__ extern char smem[];
    uint64_t input = (static_cast<int>(threadIdx.x + blockIdx.x * blockDim.x) >= size) ? 0 : value[threadIdx.x + blockIdx.x * blockDim.x];
    uint64_t* blockReduction = (uint64_t*)(smem);
    sum<uint64_t>(reinterpret_cast<uint64_t*>(output), blockReduction, input);
}

extern "C" int getGridDim(int numKeys, void*) {
    return (numKeys + 511) / 512;
}

extern "C" int getBlockDim(int, void*) {
    return 512;
}

extern "C" int getSharedMemory(int, void*) {
    return sizeof(uint64_t) * 512 / 32;
}

extern "C" int getOutputSize(int numKeys, void* args) {
    return sizeof(uint64_t) * getGridDim(numKeys, args);
}

extern "C" int useCooperativeGroups() {
    return 0;
}
