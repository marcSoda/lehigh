#include <VariadicType.h>
#include <QueryKernel.h>
#include "ml.h"
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>

extern "C" cudaError_t layerscomputeKernel(int, void* args_, void* output_, cudaStream_t stream) {
    Arguments* args = reinterpret_cast<Arguments*>(args_);
    float* output = reinterpret_cast<float*>(output_);
    runLayer1(*args, stream);
    runLayer2(*args, stream);
    copyAndFree(args, output, stream);
    return cudaSuccess;
}

extern "C" int layersgetOutputSize(int, void* args) {
    return 2 * reinterpret_cast<Arguments*>(args)->count * sizeof(float);
}

using RowMajor = cutlass::layout::RowMajor;
using ColumnMajor = cutlass::layout::ColumnMajor;
using OperatorClass = cutlass::arch::OpClassSimt;
using ArchTag = cutlass::arch::Sm70;
using Config = cutlass::gemm::device::DefaultGemmConfiguration<OperatorClass, ArchTag, float, float, float, float>;

template<typename A, typename B, typename C, typename E>
using Layer = cutlass::gemm::device::Gemm<float,
                                               A,
                                               float,
                                               B,
                                               float,
                                               C,
                                               float,
                                               OperatorClass,
                                               ArchTag,
                                               cutlass::gemm::GemmShape<128, 128, 8>,
                                               cutlass::gemm::GemmShape<32, 64, 8>,
                                               cutlass::gemm::GemmShape<1, 1, 1>,
                                               // epliogue
                                               E>;

cudaError_t init(Arguments* args, int count_, cudaStream_t stream) {
    args->count = count_;
    cudaError_t error;

    if((error = cudaMallocAsync(&args->W1, sizeof(float) * 128 * 5, stream)) != cudaSuccess) return error; 
    if((error = cudaMallocAsync(&args->W2, sizeof(float) * 128 * 2, stream)) != cudaSuccess) return error; 
    if((error = cudaMallocAsync(&args->b1, sizeof(float) * 128 * args->count, stream)) != cudaSuccess) return error; 
    if((error = cudaMallocAsync(&args->b2, sizeof(float) * 2 * args->count, stream)) != cudaSuccess) return error; 
    if((error = cudaMallocAsync(&args->x, sizeof(float) * 5 * args->count, stream)) != cudaSuccess) return error; 
    if((error = cudaMallocAsync(&args->y, sizeof(float) * 128 * args->count, stream)) != cudaSuccess) return error; 
    if((error = cudaMallocAsync(&args->z, sizeof(float) * 2 * args->count, stream)) != cudaSuccess) return error; 

    if((error = cudaMemcpyAsync(args->W1, trainedW1, sizeof(float) * 128 * 5, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return error; 
    if((error = cudaMemcpyAsync(args->W2, trainedW2, sizeof(float) * 128 * 2, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return error; 
   
    for(int i = 0; i < args->count; i++) { 
        if((error = cudaMemcpyAsync(args->b1 + i * 128, trainedb1, sizeof(float) * 128, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return error; 
        if((error = cudaMemcpyAsync(args->b2 + i * 2, trainedb2, sizeof(float) * 2, cudaMemcpyHostToDevice, stream)) != cudaSuccess) return error;
    }
    
    return cudaSuccess; 
}

cudaError_t copyAndFree(Arguments* args, float* output, cudaStream_t stream) {
    cudaError_t error;
    if((error = cudaFreeAsync(args->W1, stream)) != cudaSuccess) return error; 
    if((error = cudaFreeAsync(args->W2, stream)) != cudaSuccess) return error; 
    if((error = cudaFreeAsync(args->b1, stream)) != cudaSuccess) return error; 
    if((error = cudaFreeAsync(args->b2, stream)) != cudaSuccess) return error; 
    if((error = cudaFreeAsync(args->x, stream)) != cudaSuccess) return error; 
    if((error = cudaFreeAsync(args->y, stream)) != cudaSuccess) return error;
    if((error = cudaMemcpyAsync(output, args->z, sizeof(float) * 2 * args->count, cudaMemcpyDeviceToHost, stream)) != cudaSuccess) return error;
    return cudaFreeAsync(args->z, stream); 
}

cudaError_t runLayer1(Arguments& args, cudaStream_t stream) {
    // A <- W1 is 128 x 5
    // B <- x is 5 x count
    // C <- b1 is 128 x count

    const int Layer1M = 128;
    const int Layer1K = 5;
    const int Layer1N = args.count;

    using Layer1 = Layer<RowMajor, ColumnMajor, ColumnMajor, cutlass::epilogue::thread::LinearCombinationRelu<float, 1>>;
    Layer1::Arguments args1({Layer1M , Layer1N, Layer1K},  // Gemm Problem dimensions
                                 {args.W1, Layer1K},    // Tensor-ref for source matrix A
                                 {args.x, Layer1K},    // Tensor-ref for source matrix B
                                 {args.b1, Layer1M},    // Tensor-ref for source matrix C
                                 {args.y, Layer1M},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                 {1.0f, 1.0f}); // Scalars used in the Epilogue

    Layer1 gemm_operator1;
    
    cutlass::Status status = gemm_operator1(args1, nullptr, stream);
    if(status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    
    return cudaSuccess;
}

cudaError_t runLayer2(Arguments& args, cudaStream_t stream) {
    // b2 = sigmoid(1.0 * W2 x + 1.0 b2)
    using Layer2 = Layer<RowMajor, ColumnMajor, ColumnMajor, cutlass::epilogue::thread::LinearCombinationSigmoid<float, 1>>;

    
    Layer2 gemm_operator2;

    // A <- W2 is 2 x 128
    // B <- y is 128 x count
    // C <- b2 is 2 x count
    const int Layer2M = 2;
    const int Layer2K = 128;
    const int Layer2N = args.count;

   
    // sigmoid(W2^T * b1 + b2)
    Layer2::Arguments args2({Layer2M , Layer2N, Layer2K},  // Gemm Problem dimensions
                                 {args.W2, Layer2K},    // Tensor-ref for source matrix A
                                 {args.y, Layer2K},    // Tensor-ref for source matrix B
                                 {args.b2, Layer2M},    // Tensor-ref for source matrix C
                                 {args.z, Layer2M},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                                 {1.0f, 1.0f}); // Scalars used in the Epilogue

    auto status = gemm_operator2(args2, nullptr, stream);
    if(status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }

    return cudaSuccess;
}

