#include <VariadicType.h>
#include <QueryKernel.h>
#include <sum.h>
#include "Q2.h"
#include <tpcc.h>
#include <LSlab/LSlabMap.h>

template<typename K, typename V>
class Map {};

extern "C" __global__ void  computeKernel(int size, void* keys_, void* values_, void* args, void* output_){

    Map<void*, int> stockToSupKey; 
    Map<int, void*> supplier;
    Map<int, void*> nation;
    Map region; 



    // assume when we snapshot multiple tables we mark where each range starts
    // and ends in args

    // look through keys and see which is a stock
    // when we find a stock check the supplier key
    // look through suppliers to check the nation
    // look through the nation to check the region
    // look through the region to check if its correct

    char* output = reinterpret_cast<char*>(output_);

    extern char __shared__ smem[];

    

}

extern "C" int getGridDim(int numKeys, void* args) {
    return (numKeys + 511) / 512;
}

extern "C" int getBlockDim(int numKeys, void* args) {
    return 512;
}

extern "C" int getSharedMemory(int numKeys, void* args) {
    return getBlockDim(numKeys, args);
}

extern "C" int getOutputSize(int numKeys, void* args) {
    int gdim = getGridDim(numKeys, args);
    return 2 * 16 * gdim * sizeof(int) + 16 * gdim * sizeof(float);
}

extern "C" void postProcess(int numKeys, void* args, void* output_) {
    
}
