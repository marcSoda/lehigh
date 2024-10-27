#include <SiKV.h>
#include <tbb/concurrent_queue.h>
#include <mutex>
#include <sys/mman.h>
#include <device/Kernel.h>

#pragma once

namespace sikv {
namespace device {

template<typename T>
class HostAlloc {
public:
    HostAlloc() : pages(new tbb::concurrent_queue<void*>()), memory(new tbb::concurrent_queue<T*>()) {
        initGPU();
        CUcontext ctx;
        drvErrchk(cuCtxGetCurrent(&ctx));
        if(ctx != mainCtx) {
            SPDLOG_ERROR("No context bound to current thread");
        }
        flags = MAP_PRIVATE | MAP_ANON | MAP_POPULATE | MAP_HUGETLB;
        addpage();
    }

    void freeall() {
        void* ptr;
        int count = 0;
        while(pages->try_pop(ptr)) {
            cudaHostUnregister(ptr);
            munmap(ptr, pageSize);
            count++;
        }
        SPDLOG_ERROR("Allocated {} pages during execution", count);
        delete pages;
        delete memory;
    }

    [[nodiscard]] T* allocate() {
        T* ptr = nullptr;
        bool b = memory->try_pop(ptr);

        if(!b) {
            barrierMutex.lock();
            b = memory->try_pop(ptr);
            if(!b) {
                addpage();
                memory->try_pop(ptr);
                if(ptr == nullptr) std::terminate();
            }
            barrierMutex.unlock();
        }
        return ptr;
    }

    void free(T* ptr) {
        memory->push(ptr);
    }

    void addpage() {
        void* page = mmap(0x0, pageSize, PROT_READ | PROT_WRITE, flags, -1, 0);
        while(page == (void*)-1) {
            perror("MMap Error");
            if((flags & MAP_HUGETLB) != 0) {
                SPDLOG_ERROR("Switched to standard pages");
                flags = MAP_PRIVATE | MAP_ANON | MAP_POPULATE;
                page = mmap(0x0, pageSize, PROT_READ | PROT_WRITE, flags, -1, 0);
            } else {
                exit(1);
            }
        }
        
        if(cudaHostRegister(page, pageSize, cudaHostRegisterDefault) != cudaSuccess) exit(1);

        pages->push(page);
        for(size_t ptr = reinterpret_cast<size_t>(page); ptr + sizeof(T) < reinterpret_cast<size_t>(page) + pageSize; ptr = reinterpret_cast<size_t>(reinterpret_cast<T*>(ptr) + 1)) {
            memory->push(reinterpret_cast<T*>(ptr));
        } 
    }

private:

    int flags;

    const size_t pageSize = 2ull << 20; //1ull << 30; //2ull << 20;
    std::mutex barrierMutex;

    tbb::concurrent_queue<void*>* pages;
    tbb::concurrent_queue<T*>* memory;   
};

}
}
