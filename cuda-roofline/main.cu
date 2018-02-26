#include <cuComplex.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <sys/time.h>
#include <iomanip>
#include <iostream>
#include "../dtime.hpp"
#include "../gpu-error.h"

using namespace std;


template <typename T>
__global__ void initKernel(T* data, size_t data_len) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x) {
    data[idx] = idx;
  }
}

template <typename T, int N, int M, int BLOCKSIZE>
__global__ void testfun(T* dA, T* dB, T* dC) {
  T* sA = dA + threadIdx.x + blockIdx.x * BLOCKSIZE * M;
  T* sB = dB + threadIdx.x + blockIdx.x * BLOCKSIZE * M;

  T sum = 0;

#pragma unroll 1
  for (int i = 0; i < M; i++) {
    T a = sA[i * BLOCKSIZE];
    T b = sB[i * BLOCKSIZE];
    T v = a - b;
    for (int i = 0; i < N; i++) {
      v = v * a - b;
    }
    sum += v;
  }
  if (threadIdx.x == 0) dC[blockIdx.x] = sum;
}

int main(int argc, char** argv) {
  typedef double dtype;
  const int M = 1000;
  // PARN is a constant from the Makefile, set via -DPARN=X
  const int N = PARN;
  const int BLOCKSIZE = 256;

  int nDevices;
  GPU_ERROR(cudaGetDeviceCount(&nDevices));

#pragma omp parallel num_threads(nDevices)
  {
    GPU_ERROR(cudaSetDevice(omp_get_thread_num()));
#pragma omp barrier
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    int numBlocks;

    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, testfun<dtype, N, M, BLOCKSIZE>, BLOCKSIZE, 0));
    int blockCount = prop.multiProcessorCount * numBlocks;

    size_t data_len = (size_t)blockCount * BLOCKSIZE * M;
    dtype* dA = NULL;
    dtype* dB = NULL;
    dtype* dC = NULL;
    size_t iters = 200;

    GPU_ERROR(cudaMalloc(&dA, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dB, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dC, data_len * sizeof(dtype)));
#pragma omp barrier
    initKernel<<<blockCount, 256>>>(dA, data_len);
    initKernel<<<blockCount, 256>>>(dB, data_len);
    initKernel<<<blockCount, 256>>>(dC, data_len);
    testfun<dtype, N, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC);
    cudaDeviceSynchronize();
#pragma omp barrier

    double start = dtime();
    for (size_t iter = 0; iter < iters; iter++) {
      testfun<dtype, N, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC);
    }
    cudaDeviceSynchronize();
    double end = dtime();
    GPU_ERROR(cudaGetLastError());

#pragma omp barrier
#pragma omp critical
    {
      cout << setprecision(3) << fixed << deviceId << " " << blockCount << " blocks   " << setw(3) << N
           << " its      " << (2.0 + N * 2.0) / (2.0 * sizeof(dtype)) << " Fl/B      "
           << setprecision(0) << setw(5)
           << iters * 2 * data_len * sizeof(dtype) / (end - start) * 1.0e-9
           << " GB/s    " << setw(6)
           << iters * (2 + N * 2) * data_len / (end - start) * 1.0e-9
           << " GF\n";
    }
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
    GPU_ERROR(cudaFree(dC));
  }
  cout << "\n";
}
