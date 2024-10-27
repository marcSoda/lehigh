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
#include <random>
#include <cstddef>
#include <SiKV.h>
#include <jitify.hpp>

#define MY_TEST(x) BOOST_AUTO_TEST_CASE(x)

//#define MY_TEST(x) TEST(BOOST_TEST_MODULE, x) 

#define ASSERT(x) BOOST_TEST((x))


std::string code = " \n\
extern \"C\" __global__ void kern() { \n\
    if(threadIdx.x == 0) { \n\
       printf(\"1\\n\"); \n\
    } \n\
} \n";

MY_TEST(compilationAndRunTest) {
    static jitify::JitCache kernel_cache;
    jitify::Program program = kernel_cache.program(code);
    auto kernelInst = program.kernel("kern").instantiate();
    kernelInst.configure(1, 32).launch(nullptr);
    drvErrchk(cuCtxSynchronize());
    std::cerr << code << std::endl;
} 
