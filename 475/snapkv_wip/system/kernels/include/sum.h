#pragma once

template<typename T>
__forceinline__ __device__ void sum(T* sum, T* blockReduction, T input) {
    const int warpId = threadIdx.x / 32;
    T reg = input;
    #pragma unroll
    for(int i = 16; i > 0; i /=2) {
        T reg2 = __shfl_down_sync((~0x0), reg, i);
        reg += reg2;
    }
    if(threadIdx.x % 32 == 0) {
        blockReduction[warpId] = reg;
    }
    __syncthreads();
    if(threadIdx.x == 0) {
        #pragma unroll
        for(int i = 1; i < blockDim.x % 32; i++) {
            reg += blockReduction[i];
        }
        sum[blockIdx.x] = reg;
    }
}


