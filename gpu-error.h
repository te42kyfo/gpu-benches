#pragma once
#define GPU_ERROR(ans)                          \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    std::cerr << "GPUassert: \"" << cudaGetErrorString(code) << "\"  in "
              << file << ": " << line << "\n";
    if (abort) exit(code);
  }
}
