#include <string_view>
#include <array>

#pragma once

enum KernelType : int {
    CPU = 0,
    GPU = 1,
    TRUSTEDLIB = 2
};

enum MemType : int {
    DEVICE = 0,
    HOST = 1
};

extern "C" struct Kernel {
    int ktype;
    int inputType;
    int outputType;
    const char* name;
};


/// Pass trustedKernel_t numKeys, args, output, and stream
typedef cudaError_t (*trustedKernel_t)(int, void*, void*, cudaStream_t);

/// Pass cpuKernel_t numKeys, args, and output
typedef void (*cpuKernel_t)(int, void*, void*);

/// Pass numKeys/ranges and args
typedef int (*kernelInfo_t)(int, void*);

/* Signature of compute kernel is:
 * (int size, void* keys, void* values, void* args, void* output, unsigned long long* startRange, unsigned long long* endRange, unsigned long long)
 */

