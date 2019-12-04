#ifndef CUDART_H
#define CUDART_H
#ifdef GKLEE

void __set_device();
void __clear_device();
void __set_host();
void __clear_host();

#define cudaFree cudaFree_impl
#define cudaFreeArray cudaFreeArray_impl
#define cudaFreeHost cudaFreeHost_impl
#define cudaMalloc cudaMalloc_impl
#define cudaMallocHost cudaMallocHost_impl
#define cudaMemcpy cudaMemcpy_impl
#define cudaMemcpyPeer cudaMemcpyPeer_impl
#define cudaMemcpyPeerAsync cudaMemcpyPeerAsync_impl
#define cudaMemcpyToSymbol cudaMemcpyToSymbol_impl
#define cudaMemset cudaMemset_impl

inline cudaError_t cudaFree_impl(void *devPtr) {
    free(devPtr);
    return cudaSuccess;
}

inline cudaError_t cudaFreeArray_impl(struct cudaArray *array) {
    free(array);
    return cudaSuccess;
}

inline cudaError_t cudaFreeHost_impl(void *ptr) {
    free(ptr);
    return cudaSuccess;
}

inline cudaError_t cudaMalloc_impl(void **devPtr, size_t size) {
    __set_device();
    *devPtr = (void *)malloc(size);
    __clear_device();

    return cudaSuccess;
}

inline cudaError_t cudaMallocHost_impl(void **ptr, size_t size) {
    __set_host();
    *ptr = (void *)malloc(size);
    __clear_host();

    return cudaSuccess;
}

inline cudaError_t cudaMemcpy_impl(void *dst, const void *src, size_t count,
                                   enum cudaMemcpyKind kind) {
    memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyPeer_impl(void *dst, int dstDevice,
                                       const void *src, int srcDevice,
                                       size_t count) {
    memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemcpyPeerAsync_impl(void *dst, int dstDevice,
                                            const void *src, int srcDevice,
                                            size_t count,
                                            cudaStream_t stream = 0) {
    memcpy(dst, src, count);
    return cudaSuccess;
}

inline cudaError_t
cudaMemcpyToSymbol_impl(char *symbol, const void *src, size_t count,
                        size_t offset = 0,
                        enum cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
    memcpy(symbol + offset, src, count);
    return cudaSuccess;
}

inline cudaError_t cudaMemset_impl(void *devPtr, int value, size_t count) {
    memset(devPtr, value, count);
    return cudaSuccess;
}
#endif
#endif