#include "mltx.h"

#pragma once


extern float trainedW1[];
extern float trainedW2[];
extern float trainedb1[];
extern float trainedb2[];

extern "C" struct Arguments {
    float* W1;
    float* b1;
    float* W2;
    float* b2;
    float* x;
    float* y;
    float* z;
    int count;
};

extern cudaError_t init(Arguments* args, int count_, cudaStream_t stream);
extern cudaError_t copyAndFree(Arguments* args, float* output, cudaStream_t stream);

cudaError_t runLayer1(Arguments&, cudaStream_t);
cudaError_t runLayer2(Arguments&, cudaStream_t);

float* runExample(Arguments args);
