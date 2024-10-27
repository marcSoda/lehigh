#include <VariadicType.h>
#include <QueryKernel.h>
#include "ml.h"

Kernel k[] = {
    {KernelType::TRUSTEDLIB, MemType::DEVICE, MemType::DEVICE, "init"},
    {KernelType::GPU, MemType::DEVICE, MemType::DEVICE, "scan"},
    {KernelType::TRUSTEDLIB, MemType::HOST, MemType::HOST, "layers"},
};

extern "C" void getKernels(Kernel** kernels, int* size) {
    *size = 3;
    *kernels = k;
}

// b1 and b2 should be column major
// x, y, and z should be column major
float* runExample(Arguments args) {
    // GEMM is alpha A B + beta C where A is M by K, B is K by N, and C in M by N
   
    if(runLayer1(args, 0x0) != cudaSuccess) {
        return nullptr;
    }
    
    if(runLayer2(args, 0x0) != cudaSuccess) {
        return nullptr;
    }
    
    return args.z;
}
