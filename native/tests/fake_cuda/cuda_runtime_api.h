#pragma once
// Test-only CPU runtime shim. Never included by production/CUDA targets.
#include <cstddef>
using cudaError_t=int;
using cudaStream_t=void*;
inline constexpr cudaError_t cudaSuccess=0,cudaErrorUnknown=999;
inline cudaStream_t const cudaStreamLegacy=reinterpret_cast<void*>(1);
enum cudaMemcpyKind { cudaMemcpyHostToDevice,cudaMemcpyDeviceToHost,cudaMemcpyDeviceToDevice };
enum cudaDeviceAttr { cudaDevAttrComputeCapabilityMajor,cudaDevAttrComputeCapabilityMinor };
cudaError_t cudaGetDevice(int*);
cudaError_t cudaDeviceGetAttribute(int*,cudaDeviceAttr,int);
cudaError_t cudaMalloc(void**,std::size_t);
cudaError_t cudaFree(void*);
cudaError_t cudaMemcpy(void*,const void*,std::size_t,cudaMemcpyKind);
cudaError_t cudaMemsetAsync(void*,int,std::size_t,cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
const char* cudaGetErrorString(cudaError_t);

cudaError_t cudaMemGetInfo(std::size_t*,std::size_t*);
cudaError_t cudaSetDevice(int);

cudaError_t cudaDeviceReset();
