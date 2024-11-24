//
// Created by depaulsmiller on 9/3/20.
//

#include <LSlab/StandardSlabDefinitions.h>
#include <vector>
#include <LSlab/Slab.h>
#include <cuda_profiler_api.h>

using namespace lslab;

const int BLOCKS = 128;
const int THREADS_PER_BLOCK = 512;

int main() {

    const int size = 1000000;

    std::hash<unsigned> hfn;

    SlabUnified<unsigned long long, unsigned, BLOCKS, THREADS_PER_BLOCK> s(size);
    auto b = new BatchBuffer<unsigned long long, unsigned, BLOCKS, THREADS_PER_BLOCK>();

    s.setGPU();
    std::cerr << "Populating" << std::endl;

    unsigned gen_key = 1;

    int inserted = 0;
    while (inserted < size / 2) {
        int k = 0;
        for (; k < size / 2 - inserted && k < THREADS_PER_BLOCK * BLOCKS; ++k) {
            unsigned key = gen_key;
            gen_key++;
            b->getBatchKeys()[k] = key;
            b->getHashValues()[k] = hfn(key);
            b->getBatchRequests()[k] = REQUEST_INSERT;
        }
        for (; k < THREADS_PER_BLOCK * BLOCKS; ++k) {
            b->getBatchRequests()[k] = REQUEST_EMPTY;
        }

        std::cerr << inserted << std::endl;
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0, 1);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));

        k = 0;
        int loopCond = size / 2 - inserted;
        for (; k < loopCond && k < THREADS_PER_BLOCK * BLOCKS; ++k) {
            if (b->getBatchValues()[k] == 0)
                inserted++;
        }
    }

    std::cerr << "Populated" << std::endl;

    gpuErrchk(cudaProfilerStart());

    for (int rep = 0; rep < 10; rep++) {

        for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; ++i) {
            unsigned key = rand() / (double) RAND_MAX * (2 * size) + 1;
            b->getBatchKeys()[i] = key;
            b->getHashValues()[i] = hfn(key);
            b->getBatchRequests()[i] = REQUEST_GET;
        }


        auto start = std::chrono::high_resolution_clock::now();
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0, rep + 2);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;

        std::cout << "Standard Uniform test" << std::endl;
        std::cout << "Latency\t" << dur.count() * 1e3 << " ms" << std::endl;
        //std::cout << "Latency 2\t" << time << " ms" << std::endl;
        //std::cout << "Throughput\t" << THREADS_PER_BLOCK * BLOCKS / (time * 1e3) << " Mops" << std::endl;
    }

    std::cout << "\nNo conflict test\n";

    for (int rep = 0; rep < 10; rep++) {

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; ++i) {
            b->getBatchKeys()[i] = i + 1;
            b->getHashValues()[i] = (i + 1) / 32;
            b->getBatchRequests()[i] = REQUEST_GET;
        }
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0, 12 + rep);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;

        std::cout << "Latency\t" << dur.count() * 1e3 << " ms" << std::endl;
        //std::cout << "Latency 2\t" << time << " ms" << std::endl;
        //std::cout << "Throughput\t" << THREADS_PER_BLOCK * BLOCKS / (time * 1e3) << " Mops" << std::endl;
        std::cout << "Arrival rate handled\t" << THREADS_PER_BLOCK * BLOCKS / (dur.count() * 1e6) << " Mops"
                  << std::endl;

    }
    gpuErrchk(cudaProfilerStop());
    delete b;
}
