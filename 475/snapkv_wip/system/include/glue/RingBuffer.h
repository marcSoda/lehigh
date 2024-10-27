#include <SiKV.h>
#include <atomic>

#pragma once

namespace sikv {
namespace glue {

/// Single producer single consumer
template<typename K, typename V>
struct LogBuffer {
public:
    RingBuffer(uint32_t size_) : buf(new T[size_]), size(size_), end(0), start(0) {
    
    }

    RingBuffer() : RingBuffer(10000) {}

    ~RingBuffer() {
        delete[] buf;
    }

    bool try_push(T& val) {
        uint32_t found = end;
        uint32_t foundStart = start;

        uint32_t foundp1modsize = found + 1;
        if(foundp1modsize >= size) {
            foundp1modsize -= size;
        }

        if(foundp1modsize == foundStart) {
            return false;
        }

        buf[found] = std::move(val);
        end.store(foundp1modsize, std::memory_order_release);

        return true;
    }

    bool try_pop(T& val) {
        uint32_t found = end.load(std::memory_order_relaxed);
        uint32_t foundStart = start.load(std::memory_order_acquire);

        if(found == foundStart) {
            return false;
        }

        val = buf[found];

        uint32_t foundp1modsize = foundStart + 1;
        if(foundp1modsize >= size) {
            foundp1modsize -= size;
        }

        start.store(foundp1modsize, std::memory_order_release);

        return true;
    }
    
private:
    T* buf;
    uint32_t size;

    std::atomic_uint32_t end;
    std::atomic_uint32_t start;
};

}
}
