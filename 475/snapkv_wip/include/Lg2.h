#include <cstdint>

#pragma once

namespace sikv {

namespace arith {

constexpr int clz(uint64_t x) {
    unsigned long long mask = 1ull << 63;
    for(int i = 0; i < 64; i++) {
        unsigned long long res = x & mask; // if res is != 0 it means bit 63 - i is set and there are i leading 0s
        if(res != 0) {
            return i;
        }
        mask >>= 1ull; 
    }
    return 64;
}

template<uint64_t N>
struct FloorLog2 {
    constexpr const static int value = 63 - clz(N);
};

}

}
