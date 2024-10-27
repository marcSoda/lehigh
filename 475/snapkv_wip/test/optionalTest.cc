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

#define BOOST_TEST_MODULE clzTest
#include <boost/test/included/unit_test.hpp>
#include <iostream>
#include <optional>
#include <random>
#include <optional.h>

#define MY_TEST(x) BOOST_AUTO_TEST_CASE(x)

//#define MY_TEST(x) TEST(BOOST_TEST_MODULE, x) 

#define ASSERT(x) BOOST_TEST((x))


MY_TEST(TxTest) {
    spdlog::set_level(spdlog::level::trace);

    cuda::std::optional<int> i = 10;

    ASSERT(*i == 10 && i != cuda::std::nullopt);
    
    i = 20;

    ASSERT(*i == 20 && i != cuda::std::nullopt);

    i = cuda::std::nullopt;

    ASSERT(i == cuda::std::nullopt);
    
    cuda::std::optional<int> j;

    ASSERT(j == cuda::std::nullopt);

    

}
