#include <iostream>

#pragma once

#ifdef __NVCC__
#define GPU_ERROR(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in "
              << file << ": " << line << "\n";
    if (abort)
      exit(code);
  }
}
#elif defined __HIP__
#define GPU_ERROR(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != hipSuccess) {
    std::cerr << "GPUassert: \"" << hipGetErrorString(code) << "\"  in " << file
              << ": " << line << "\n";
    if (abort)
      exit(code);
  }
}
#endif
