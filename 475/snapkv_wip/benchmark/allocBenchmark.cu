/*
 * Copyright (c) 2021-2022 dePaul Miller (dsm220@lehigh.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <iostream>
#include <GroupAllocator/groupallocator>
#include <benchmark/benchmark.h>
#include <random>

static void BM_cuUMAlloc(benchmark::State& state) {
    unsigned long count = 0;
    for(auto _: state) {
        char* buffer;
        cudaMallocManaged(&buffer, sizeof(char) * 256);    
        cudaFree(buffer);
        count++;    
    }
    state.counters["Rate"] = benchmark::Counter(256 * count, benchmark::Counter::kIsRate);
}

static void BM_cuAlloc(benchmark::State& state) {
    unsigned long count = 0;
    for(auto _: state) {
        char* buffer;
        cudaMalloc(&buffer, sizeof(char) * 256);    
        cudaFree(buffer);
        count++;    
    }
    state.counters["Rate"] = benchmark::Counter(256 * count, benchmark::Counter::kIsRate);
}

static void BM_cuAllocAndInit(benchmark::State& state) {
    unsigned long count = 0;
    char buf[256];
    for(auto _: state) {
        char* buffer;
        cudaMalloc(&buffer, sizeof(char) * 256);    
        cudaMemcpy(buffer, buf, sizeof(char) * 256, cudaMemcpyHostToDevice);
        cudaFree(buffer);
        count++;    
    }
    state.counters["Rate"] = benchmark::Counter(256 * count, benchmark::Counter::kIsRate);
}

static void BM_cuInit(benchmark::State& state) {
    unsigned long count = 0;
    char buf[256];
    char* buffer;
    cudaMalloc(&buffer, sizeof(char) * 256);    
    for(auto _: state) {
        cudaMemcpy(buffer, buf, sizeof(char) * 256, cudaMemcpyHostToDevice);
        count++;    
    }
    cudaFree(buffer);
    state.counters["Rate"] = benchmark::Counter(256 * count, benchmark::Counter::kIsRate);
}

groupallocator::GroupAllocator* alloc;

static void BM_groupAlloc(benchmark::State& state) {
    unsigned long count = 0;

    if(state.thread_index() == 0) {
        alloc = new groupallocator::GroupAllocator(0, 1 << 21);
    }

    for(auto _: state) {
        char* buffer;
        alloc->allocate(&buffer, sizeof(char) * 256, false);    
        alloc->free(buffer);
        count++; 
    }
    state.counters["Rate"] = benchmark::Counter(256 * count, benchmark::Counter::kIsRate);
    
    if(state.thread_index() == 0) {
        alloc->freeall();
        delete alloc;
    }

}

BENCHMARK(BM_cuUMAlloc)->ThreadRange(1, 12);
BENCHMARK(BM_groupAlloc)->ThreadRange(1, 12);
BENCHMARK(BM_cuAlloc)->ThreadRange(1, 12);
BENCHMARK(BM_cuAllocAndInit)->ThreadRange(1, 12);
BENCHMARK(BM_cuInit)->ThreadRange(1, 12);

BENCHMARK_MAIN();
