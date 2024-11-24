/**
 * @author dePaul Miller
 */
#include <atomic>
#include <future>
#include <mutex>
#include <thread>
#include <type_traits>
#include "LSlab.h"

#include "stdrequestHandler.cuh"

#pragma once

namespace lslab {

#define DEFAULT_SHOULD_USE_HOST true

LSLAB_HOST int getBlocks() noexcept {
    cudaDeviceProp prop;
    prop.multiProcessorCount = 68;
    gpuAssert_slab(cudaGetDeviceProperties(&prop, 0), __FILE__, __LINE__, false);
    return 2 * prop.multiProcessorCount;
}

//const int BLOCKS = getBlocks();
//const int THREADS_PER_BLOCK = 512;

template<typename T>
struct UniqueHostDevicePtr {

    UniqueHostDevicePtr() : h(nullptr), k(nullptr), s(0) {

    }

    UniqueHostDevicePtr(size_t size) : h(nullptr), k(nullptr), s(size) {
        gpuErrchk(cudaMalloc(&k, size));
        gpuErrchk(cudaMallocHost(&h, size));
    }


    UniqueHostDevicePtr(const UniqueHostDevicePtr<T> &) = delete;

    UniqueHostDevicePtr(UniqueHostDevicePtr<T> &&rhs) noexcept {
        this->h = rhs.h;
        this->k = rhs.k;
        this->s = rhs.s;
        rhs.h = nullptr;
        rhs.k = nullptr;
        rhs.s = 0;
    }

    UniqueHostDevicePtr<T> &operator=(UniqueHostDevicePtr<T> &&rhs) noexcept {
        this->h = rhs.h;
        this->k = rhs.k;
        this->s = rhs.s;
        rhs.h = nullptr;
        rhs.k = nullptr;
        rhs.s = 0;
        return *this;
    }


    ~UniqueHostDevicePtr() {
        //std::cerr << "Delete called on " << (void *) k << " " << (void *) h << std::endl;
        if (k) gpuErrchk(cudaFree(k));
        if (h) gpuErrchk(cudaFreeHost(h));
    }

    T *getHost() {
        return h;
    }

    T *getDevice() {
        return k;
    }

    void moveToGPUAsync(cudaStream_t stream = cudaStreamDefault) {
        gpuErrchk(cudaMemcpyAsync(k, h, s, cudaMemcpyHostToDevice, stream));
    }

    void moveToCPUAsync(cudaStream_t stream = cudaStreamDefault) {
        gpuErrchk(cudaMemcpyAsync(h, k, s, cudaMemcpyDeviceToHost, stream));
    }

private:
    T *h;
    T *k;
    size_t s;
};

template<typename K, typename V, bool B = false>
struct AddExtra {
    AddExtra() = default;

    AddExtra(size_t) {}

    AddExtra(AddExtra<K, V, true> &&rhs) {}
};

template<typename K, typename V>
struct AddExtra<K, V, true> {
    AddExtra() = default;

    AddExtra(size_t s) : batchKeys(s * sizeof(K)), batchValues(s * sizeof(V)), batchRequests(s * sizeof(int)),
                         hashValues(s * sizeof(unsigned)) {
        std::cerr << "Allocated memory in host and device\n";
    }

    AddExtra(AddExtra<K, V, true> &&rhs) noexcept {
        batchKeys = std::move(rhs.batchKeys);
        batchValues = std::move(rhs.batchValues);
        batchRequests = std::move(rhs.batchRequests);
        hashValues = std::move(rhs.hashValues);
    }

    AddExtra<K, V, true> &operator=(AddExtra<K, V, true> &&rhs) noexcept {
        batchKeys = std::move(rhs.batchKeys);
        batchValues = std::move(rhs.batchValues);
        batchRequests = std::move(rhs.batchRequests);
        hashValues = std::move(rhs.hashValues);
        return *this;
    }

    UniqueHostDevicePtr<K> batchKeys;
    UniqueHostDevicePtr<V> batchValues;
    UniqueHostDevicePtr<int> batchRequests;
    UniqueHostDevicePtr<unsigned> hashValues;
};

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK, bool UseHost = DEFAULT_SHOULD_USE_HOST>
class SlabUnified;

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK, bool UseHost = DEFAULT_SHOULD_USE_HOST>
class BatchBuffer;

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK, bool UseHost = false>
struct AllocateBuffers {
    inline void operator()(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *b) {
        std::cerr << "Allocating Buffers\n";
        b->bufferGAlloc->allocate(&b->batchKeys,
                                  BLOCKS * THREADS_PER_BLOCK * sizeof(K), false);
        b->bufferGAlloc->allocate(&b->batchValues,
                                  BLOCKS * THREADS_PER_BLOCK * sizeof(V), false);
        b->bufferGAlloc->allocate(&b->hashValues,
                                  BLOCKS * THREADS_PER_BLOCK * sizeof(unsigned), false);
        b->bufferGAlloc->allocate(&b->batchRequests,
                                  BLOCKS * THREADS_PER_BLOCK * sizeof(int), false);

        b->batchKeys_k = b->batchKeys;
        b->batchValues_k = b->batchValues;
        b->batchRequests_k = b->batchRequests;
        b->hashValues_k = b->hashValues;
    }
};

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK>
struct AllocateBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, true> {
    inline void operator()(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, true> *b) {
        std::cerr << "Allocating Buffers On Host Device\n";
        b->newMemory = std::move(AddExtra<K, V, true>(BLOCKS * THREADS_PER_BLOCK));
        b->batchKeys = b->newMemory.batchKeys.getHost();
        b->batchValues = b->newMemory.batchValues.getHost();
        b->batchRequests = b->newMemory.batchRequests.getHost();
        b->hashValues = b->newMemory.hashValues.getHost();

        b->batchKeys_k = b->newMemory.batchKeys.getDevice();
        b->batchValues_k = b->newMemory.batchValues.getDevice();
        b->batchRequests_k = b->newMemory.batchRequests.getDevice();
        b->hashValues_k = b->newMemory.hashValues.getDevice();

    }
};

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK, bool UseHost = false>
struct MoveBuffers {
    static inline void toCPU(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, false> *b, cudaStream_t stream) {
        b->bufferGAlloc->moveToDevice(cudaCpuDeviceId, stream);

    }

    static inline void toGPU(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, false> *b, cudaStream_t stream) {
        b->bufferGAlloc->moveToDevice(b->_gpu, stream);
    }

};

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK>
struct MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, true> {
    static inline void toCPU(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, true> *b, cudaStream_t stream) {
        b->newMemory.batchValues.moveToCPUAsync(stream);
    }

    static inline void toGPU(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, true> *b, cudaStream_t stream) {
        b->newMemory.batchKeys.moveToGPUAsync(stream);
        b->newMemory.batchValues.moveToGPUAsync(stream);
        b->newMemory.batchRequests.moveToGPUAsync(stream);
        b->newMemory.hashValues.moveToGPUAsync(stream);
    }
};


template<typename K, typename V>
class Slab {
public:

protected:
    SlabCtx<K, V> *slab{};
    WarpAllocCtx<K, V> ctx;
    int _gpu{};
    int mapSize{};
    std::thread *handler{};
    std::atomic<bool> *signal{};
    std::mutex mtx;
    int position{};
};

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK, bool UseHost>
class BatchBuffer {
public:
    BatchBuffer() {
        bufferGAlloc = new groupallocator::GroupAllocator(2, 4096);
        allocateBuffersFn(this);
    }

    ~BatchBuffer() {
        delete bufferGAlloc;
    }

    inline K *getBatchKeys() {
        return batchKeys;
    }

    inline V *getBatchValues() {
        return batchValues;
    }

    inline int *getBatchRequests() {
        return batchRequests;
    }

    inline unsigned *getHashValues() {
        return hashValues;
    }

private:
    K *batchKeys{};
    V *batchValues{};
    int *batchRequests{};
    unsigned *hashValues{};
    K *batchKeys_k{};
    V *batchValues_k{};
    int *batchRequests_k{};
    unsigned *hashValues_k{};

    groupallocator::GroupAllocator *bufferGAlloc;

    AddExtra<K, V, UseHost> newMemory;
    AllocateBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> allocateBuffersFn;

    friend class AllocateBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>;

    friend class MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>;

    friend class SlabUnified<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>;
};

template<typename K, typename V, int BLOCKS, int THREADS_PER_BLOCK, bool UseHost>
class SlabUnified : public Slab<K, V> {
public:
    SlabUnified(int size) : SlabUnified(size, 0) {}

    SlabUnified() : slabGAlloc(nullptr),
                    allocGAlloc(nullptr) {}

    SlabUnified(int size, int gpu) {
        gpuErrchk(cudaSetDevice(gpu));

        slabGAlloc = new groupallocator::GroupAllocator(0, 4096);
        allocGAlloc = new groupallocator::GroupAllocator(1, 4096);
        this->slab = setUpGroup<K, V>(*slabGAlloc, size, gpu, cudaStreamDefault);
        this->ctx = setupWarpAllocCtxGroup<K, V>(*allocGAlloc, THREADS_PER_BLOCK, BLOCKS,
                                                 gpu, cudaStreamDefault);

        this->_gpu = gpu;
        this->mapSize = size;
    }

    SlabUnified(SlabUnified<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> &&other) noexcept {
        gpuErrchk(cudaSetDevice(other._gpu));

        this->_stream = other._stream;
        other._stream = nullptr;

        slabGAlloc = other.slabGAlloc;
        other.slabGAlloc = nullptr;

        allocGAlloc = other.allocGAlloc;
        other.allocGAlloc = nullptr;

        this->slab = other.slab;
        other.slab = nullptr;
        this->batchKeys = other.batchKeys;
        other.batchKeys = nullptr;
        this->batchValues = other.batchValues;
        other.batchValues = nullptr;
        this->hashValues = other.hashValues;
        other.hashValues = nullptr;

        this->batchRequests = other.batchRequests;
        other.batchRequests = nullptr;

        this->ctx = other.ctx;
        other.ctx = WarpAllocCtx<K, V>();
        this->_gpu = other._gpu;
        this->mapSize = other.mapSize;
    }


    ~SlabUnified() {
        delete slabGAlloc;
        delete allocGAlloc;
    }

    SlabUnified<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> &operator=(SlabUnified<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> &&other) noexcept {
        gpuErrchk(cudaSetDevice(other._gpu));

        slabGAlloc = other.slabGAlloc;
        other.slabGAlloc = nullptr;

        allocGAlloc = other.allocGAlloc;
        other.allocGAlloc = nullptr;

        this->slab = other.slab;
        other.slab = nullptr;

        this->ctx = other.ctx;
        other.ctx = WarpAllocCtx<K, V>();
        this->_gpu = other._gpu;
        this->mapSize = other.mapSize;

        return *this;
    }

    /**
     * Takes in THREADS_PER_BLOCK * BLOCKS sized arrays of keys, values, requests, and hashes
     * Sets values as response
     * value index is set to the value on a GET or EMPTY<V>::value if there is no value
     * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
     * @param keys
     * @param values
     * @param requests
     * @param hashes
     */
    void batch(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer) {
        batch(buffer, BLOCKS, THREADS_PER_BLOCK, cudaStreamDefault);
    }


    /**
     * Takes in THREADS_PER_BLOCK * BLOCKS sized arrays of keys, values, requests, and hashes
     * Sets values as response
     * value index is set to the value on a GET or EMPTY<V>::value if there is no value
     * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
     * @param keys
     * @param values
     * @param requests
     * @param hashes
     */
    void batch(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, unsigned blocks,
               unsigned threads_per_block, cudaStream_t stream, uint64_t iter) {

        gpuErrchk(cudaSetDevice(this->_gpu));

        MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>::toGPU(buffer, stream);
        gpuErrchk(cudaStreamSynchronize(stream));

        //std::cerr << "Moved to device " << std::endl;

        requestHandlerAPI<K, V><<<blocks, threads_per_block, 0, stream>>>(
                this->slab->slabs, this->slab->num_of_buckets, buffer->batchKeys_k, buffer->batchValues_k,
                buffer->hashValues_k,
                buffer->batchRequests_k,
                this->ctx, iter);
        gpuErrchk(cudaStreamSynchronize(stream));

        //std::cerr << "Request handler done " << std::endl;

        MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>::toCPU(buffer, stream);
        gpuErrchk(cudaStreamSynchronize(stream));
        //std::cerr << "Moved to cpu " << std::endl;
    }

    /**
     * Takes in THREADS_PER_BLOCK * BLOCKS sized arrays of keys, values, requests, and hashes
     * Sets time to the milliseconds the kernel takes to run. Sets values as response.
     * Value index is set to the value on a GET or EMPTY<V>::value if there is no value
     * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
     *
     * @param keys
     * @param values
     * @param requests
     * @param hashes
     * @param time
     */
    void batch(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, float &time, uint64_t iter) {
        batch(buffer, time, BLOCKS, THREADS_PER_BLOCK, cudaStreamDefault, iter);
    }


    /**
     * Takes in THREADS_PER_BLOCK * BLOCKS sized arrays of keys, values, requests, and hashes
     * Sets time to the milliseconds the kernel takes to run. Sets values as response.
     * Value index is set to the value on a GET or EMPTY<V>::value if there is no value
     * It is set to (V)1 on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
     *
     * @param keys
     * @param values
     * @param requests
     * @param hashes
     * @param time
     */
    void batch(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, float &time, unsigned blocks,
               unsigned threads_per_block, cudaStream_t stream, uint64_t iter) {

        gpuErrchk(cudaSetDevice(this->_gpu));

        cudaEvent_t startEvent;
        cudaEvent_t endEvent;

        gpuErrchk(cudaEventCreate(&startEvent));
        gpuErrchk(cudaEventCreate(&endEvent));

        MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>::toGPU(buffer, stream);
        gpuErrchk(cudaStreamSynchronize(stream));

        gpuErrchk(cudaEventRecord(startEvent, stream));

        //std::cerr << "Moved to device " << std::endl;

        requestHandlerAPI<K, V><<<blocks, threads_per_block, 0, stream>>>(
                this->slab->slabs, this->slab->num_of_buckets, buffer->batchKeys_k, buffer->batchValues_k,
                buffer->hashValues_k,
                buffer->batchRequests_k,
                this->ctx,
                iter);

        gpuErrchk(cudaEventRecord(endEvent, stream));


        gpuErrchk(cudaEventSynchronize(endEvent));

        MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>::toCPU(buffer, stream);

        gpuErrchk(cudaEventElapsedTime(&time, startEvent, endEvent));

        gpuErrchk(cudaStreamSynchronize(stream));
        //std::cerr << "Moved to cpu " << std::endl;

        gpuErrchk(cudaEventDestroy(startEvent));
        gpuErrchk(cudaEventDestroy(endEvent));

    }

    inline void setGPU() {
        gpuErrchk(cudaSetDevice(this->_gpu));
    }

    /**
     * Set GPU before this call.
     * @param buffer
     * @param stream
     */
    inline void moveBufferToGPU(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, cudaStream_t stream) {
        MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>::toGPU(buffer, stream);
    }

    /**
     * Set GPU before this.
     * @param buffer
     * @param stream
     */
    inline void moveBufferToCPU(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, cudaStream_t stream) {
        MoveBuffers<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost>::toCPU(buffer, stream);
    }

    /**
     * Need to set device to this GPU first by calling setGPU. Then moving data to GPU. Data should be moved after.
     * This is entirely asynchronous.
     * @param buffer
     * @param time
     * @param blocks
     * @param threads_per_block
     * @param stream
     */
    void
    diy_batch(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, unsigned blocks, unsigned threads_per_block, cudaStream_t stream, uint64_t iter) {
        requestHandlerAPI<K, V><<<blocks, threads_per_block, 0, stream>>>(
                this->slab->slabs, this->slab->num_of_buckets, buffer->batchKeys_k, buffer->batchValues_k,
                buffer->hashValues_k,
                buffer->batchRequests_k,
                this->ctx,
                iter);
    }

    void
    diy_batch_traditional(BatchBuffer<K, V, BLOCKS, THREADS_PER_BLOCK, UseHost> *buffer, unsigned blocks, unsigned threads_per_block, cudaStream_t stream, uint64_t iter) {
        requestHandlerAPI<K, V><<<blocks, threads_per_block, 0, stream>>>(
                this->slab->slabs, this->slab->num_of_buckets, buffer->batchKeys_k, buffer->batchValues_k,
                buffer->hashValues_k,
                buffer->batchRequests_k,
                this->ctx,
                iter);
    }
    size_t getTable(K*& keys, V*& values, unsigned blocks,
               unsigned threads_per_block, cudaStream_t stream) {

        K* keys_k;
        V* values_k;
        size_t containsGuess = 31 * this->slab->num_of_buckets / 2;
        unsigned long long* size_k;
        int* fail;

        gpuErrchk(cudaSetDevice(this->_gpu));

        gpuErrchk(cudaMallocManaged(&keys_k, sizeof(K) * containsGuess));
        gpuErrchk(cudaMallocManaged(&values_k, sizeof(V) * containsGuess));
        gpuErrchk(cudaMallocManaged(&size_k, sizeof(unsigned long long)));
        gpuErrchk(cudaMallocManaged(&fail, sizeof(int)));
        *fail = 0;
        *size_k = 0;

        dumpTable<K, V><<<1, 32, 0, stream>>>(
                this->slab->slabs, this->slab->num_of_buckets, keys_k, values_k,
                containsGuess, size_k, fail, this->ctx);

        gpuErrchk(cudaStreamSynchronize(stream));

        while(*fail == 1) {
            std::cerr << "Failed to allocate memory" << std::endl;
            gpuErrchk(cudaFree(keys_k));
            gpuErrchk(cudaFree(values_k));
            containsGuess = containsGuess * 2 + 1;
            gpuErrchk(cudaMallocManaged(&keys_k, sizeof(K) * containsGuess));
            gpuErrchk(cudaMallocManaged(&values_k, sizeof(V) * containsGuess));

            *fail = 0;
            *size_k = 0;

            dumpTable<K, V><<<1, 32, 0, stream>>>(
                this->slab->slabs, this->slab->num_of_buckets, keys_k, values_k,
                containsGuess, size_k, fail, this->ctx);

            gpuErrchk(cudaStreamSynchronize(stream));

        }

        size_t written = *size_k;

        std::cerr << written << " were written" << std::endl;

        keys = new K[written];
        values = new V[written];

        memcpy(keys, keys_k, sizeof(K) * written);
        memcpy(values, values_k, sizeof(V) * written);

        gpuErrchk(cudaFree(keys_k));
        gpuErrchk(cudaFree(values_k));
        gpuErrchk(cudaFree(size_k));
        gpuErrchk(cudaFree(fail));

        //std::cerr << "Moved to cpu " << std::endl;

        return written;

    }

private:
    groupallocator::GroupAllocator *slabGAlloc;
    groupallocator::GroupAllocator *allocGAlloc;


};

}
