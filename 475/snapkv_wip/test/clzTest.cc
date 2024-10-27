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
#define BOOST_TEST_MODULE clzTest
#include <boost/test/included/unit_test.hpp>
//#include <gtest/gtest.h>
#include <Lg2.h>
#include <random>
#include <spdlog/spdlog.h>

#define MY_TEST(x) BOOST_AUTO_TEST_CASE(x)

//#define MY_TEST(x) TEST(BOOST_TEST_MODULE, x) 

#define ASSERT(x) BOOST_TEST((x))

using namespace sikv;

constexpr void clzConstexprTest() {
    static_assert(arith::clz(0) == 64, "clz 0 is 64");
    static_assert(arith::clz(1ull << 10) == 53, "clz 2^10 is 53");
    static_assert(arith::clz(1ull << 22) == 41, "clz 2^22 is 41");
}

MY_TEST(clzTest) {
    
    clzConstexprTest();

    ASSERT(arith::clz(0) == __builtin_clzll(0)); 
        
    //SPDLOG_ERROR("not true for 0");
    
    for(int i = 0; i < 64; i++) {
        unsigned long long x = 1ull << i;
        ASSERT(arith::clz(x) == __builtin_clzll(x)); 
        //SPDLOG_ERROR("Not true for {} x is {} constexpr is {} builtin is {}", i, x, arith::clz(x), __builtin_clzll(x));
    }

    for(int i = 0; i < 100; i++) {
        unsigned long long x = rand();
        ASSERT(arith::clz(x) == __builtin_clzll(x)); //<< " not true for x: x is " << x << " constexpr is " << arith::clz(x) << " builtin is " << __builtin_clzll(x);
    }

}

MY_TEST(ln2Test) {

    int val = arith::FloorLog2<1000>::value;

    ASSERT(val == (63 - __builtin_clzll(1000)));
   
    static_assert(arith::FloorLog2<1ull << 10>::value == 10, "exact test works");

}
