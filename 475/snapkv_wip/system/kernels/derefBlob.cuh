
template<typename T>
__forceinline__ __device__ T* derefBlobDevice(char* in, size_t offset) {
    return reinterpret_cast<T*>(in + offset);
}

template<typename T>
__global__ void derefBlobKernel(size_t size, char** in, T* out, size_t offset) {
    if(threadIdx.x + blockDim.x * blockIdx.x < size) {
        char* data = in[threadIdx.x + blockIdx.x * blockDim.x];
        out[threadIdx.x + blockIdx.x * blockDim.x] = *derefBlobDevice<T>(data, offset);
    }
}
