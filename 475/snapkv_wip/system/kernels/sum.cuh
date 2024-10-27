
template<typename T>
__forceinline__ __device__ void sumDevice(size_t size, T* sum, T* blockReduction, T input) {
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

template<typename T>
__global__ void sumKernel(size_t size, T* sum, T* data) {
    __shared__ extern char smem[];
    T input = (threadIdx.x + blockIdx.x * blockDim.x >= size) ? T{} : data[threadIdx.x + blockIdx.x * blockDim.x];
    T* blockReduction = (T*)(smem);
    sumDevice<T>(size, sum, blockReduction, input);
}
