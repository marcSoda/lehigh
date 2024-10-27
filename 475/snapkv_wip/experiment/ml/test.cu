#include "ml.h"
#include <iostream>
#include <SiKV.h>

#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}

float testX[] = {2.625315003598004,
    -1.403227645910433,
    2.6251738277475365,
    -1.4032859842224563,
    955.7438556748073
};

float testXBatch[] = {2.625315003598004,
    -1.403227645910433,
    2.6251738277475365,
    -1.4032859842224563,
    955.7438556748073, // end 0
    -0.23212437221165486,
    -0.9524087756789281,
    -0.23215288645113893,
    -0.9523293485780118,
    524.8956174958913, // end 1
    -3.119996441972598,
    -0.4746844215721043,
    -3.119972761981784,
    -0.4748239970924448,
    901.7378739507286, // end 2
    -2.0995241636538555,
    0.2854217002950413,
    -2.099639567501381,
    0.2855115220242683,
    789.8853511075672, // end 3
    0.5969287599786988,
    0.9493159598019668,
    0.5968555299110434,
    0.9492437811372357,
    601.9328591147396 // end 4
};

int main() {

    Arguments args;

    init(&args, 1, 0x0);

    gpuErrchk(cudaMemcpyAsync(args.x, testX, sizeof(float) * 5, cudaMemcpyHostToDevice, 0x0));
    
    auto f = runExample(args);

    if(f == nullptr) {
        std::cerr << "Got nullptr - error" << std::endl;
    }

    gpuErrchk(cudaDeviceSynchronize());

    float intermediate[128] = {};
    float intermediate_k[128];

    for(int i = 0; i < 128; i++) {
        for(int j = 0; j < 5; j++) {
            intermediate[i] += trainedW1[i * 5 + j] * testX[j];
        }
        intermediate[i] += trainedb1[i];
        intermediate[i] = std::max(0.0f, intermediate[i]);
    }

    gpuErrchk(cudaMemcpy(intermediate_k, args.y, sizeof(float) * 128, cudaMemcpyDeviceToHost));
    
    for(int i = 0; i < 128 ;i++) {
        std::cout << intermediate_k[i] << "\tvs\t" << intermediate[i] << std::endl;
    }
    std::cout << std::endl;

    float expected[2] = {};

    for(int i = 0; i < 2; i++) {
        for(int j = 0; j < 128; j++) {
            expected[i] += trainedW2[i * 128 + j] * intermediate[j];
        }
        expected[i] += trainedb2[i];
        expected[i] = 1 / (1 + expf(-expected[i]));
    }

    std::cout << expected[0] << std::endl;
    std::cout << expected[1] << std::endl;

    float res[2] = {};

    copyAndFree(&args, res, 0x0);
    gpuErrchk(cudaDeviceSynchronize());

    std::cout << res[0] << std::endl;
    std::cout << res[1] << std::endl;

    float tolerance = 1e-3;
    if(abs(res[0] - expected[0]) > tolerance || abs(res[1] - expected[1]) > tolerance) {
        return 1;
    }


    init(&args, 5, 0x0);
    
    gpuErrchk(cudaMemcpy(args.x, testXBatch, sizeof(float) * 5 * 5, cudaMemcpyHostToDevice));

    f = runExample(args);

    if(f == nullptr) {
        std::cerr << "Got nullptr - error" << std::endl;
    }

    float* result = new float[128 * 5];
    copyAndFree(&args, result, 0x0);
    gpuErrchk(cudaDeviceSynchronize());

    for(int i = 0; i < 5; i++) {
        std::cout << "Result: " << std::endl;
        for(int j = 0; j < 2; j++) {
            std::cout << result[j + i * 2] << std::endl;
        }
        std::cout << std::endl;
    }

    delete[] result;

    return 0;
}
