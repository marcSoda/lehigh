/*
 * Copyright (c) 2020-2021 dePaul Miller (dsm220@lehigh.edu)
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
#include <SiKVHTAP.h>
#include <benchmark/benchmark.h>
#include <memory>
#include <random>
#include <atomic>
#include <mlprofile.h>

using namespace sikv;

class InferenceFixture : public ::benchmark::Fixture {
public:

    InferenceFixture() {}

    void SetUp(const ::benchmark::State& state) {
        SetUp_(state);
    }

    void SetUp(::benchmark::State& state) {
        SetUp_(state);
    }

    void SetUp_(const ::benchmark::State& state) {}
    
    void TearDown(const ::benchmark::State&) {}

    void TearDown(::benchmark::State&) {}

};

BENCHMARK_DEFINE_F(InferenceFixture, FullPipeline)(benchmark::State& state) {
    int ops = 0;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for(auto _: state) {
        Arguments args;
        init(&args, 1, 0x0);
        float x[] = {1.0f, 1.0f, 1.0f, 1.0f, 0x0}; 
        float z[] = {0, 0};
        cudaMemcpyAsync(args.x, x, sizeof(float) * 5, cudaMemcpyHostToDevice, stream);
        runLayer1(args, stream);
        runLayer2(args, stream);
        copyAndFree(&args, z, stream);
        if(cudaStreamSynchronize(stream) != cudaSuccess) {
            exit(3);
        }
        ops++;
    }

    cudaStreamDestroy(stream);
    state.counters["Throughput"] = benchmark::Counter(ops, benchmark::Counter::kIsRate);
}

BENCHMARK_REGISTER_F(InferenceFixture, FullPipeline)->UseRealTime()->ThreadRange(1, 12);

BENCHMARK_MAIN();
