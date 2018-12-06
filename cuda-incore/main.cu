#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../gpu_clock.cuh"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>
#include <map>

using namespace std;

template <typename T> __global__ void initKernel(T *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }
}

template <typename T, int N, int M>
__global__ void dependent_FMA_mixed(T p, T *A, int iters) {
#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    T t[M];
#pragma unroll
    for (int m = 0; m < M; m++) {
      t[m] = p + threadIdx.x + iter + m;
    }
#pragma unroll
    for (int n = 0; n < N / M; n++) {
#pragma unroll
      for (int m = 0; m < M; m++) {
        t[m] = t[m] * (T)0.9 + (T)0.5;
      }
    }
#pragma unroll
    for (int m = 0; m < M; m++) {
      if (t[m] > (T)22313.0) {
        A[0] = t[m];
      }
    }
  }
}

template <typename T, int N, int M>
__global__ void dependent_FMA_seperated(T p, T *A, int iters) {

  for (int iter = 0; iter < iters; iter++) {
#pragma unroll
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;
      for (int n = 0; n < N; n++) {
        t = t * (T)0.9 + (T)0.5;
      }
      if (t > (T)22313.0) {
        A[0] = t;
      }
    }
  }
}

template <typename T, int N, int M>
__global__ void dependent_div_seperated(T p, T *A, int iters) {

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;

      for (int n = 0; n < N; n++) {
        t = 0.1 / (t + 0.2);
      }

      A[threadIdx.x + iter] = t;
    }
  }
}

template <typename T, int N, int M>
__global__ void dependent_sqrt_seperated(T p, T *A, int iters) {

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {

    for (int m = 0; m < M; m++) {
      T t = p + threadIdx.x + iter + m;

      for (int n = 0; n < N; n++) {
        t = sqrt(t + 0.2);
      }

      A[threadIdx.x + iter] = t;
    }
  }
}

template <typename T, int N, int M> double measure(int warpCount) {
  const int iters = 10000;
  const int blockSize = 32 * warpCount;
  const int blockCount = 1;

  MeasurementSeries time;

  T *dA;
  GPU_ERROR(cudaMalloc(&dA, iters * 2 * sizeof(T)));
  initKernel<<<52, 256>>>(dA, iters * 2);
  GPU_ERROR(cudaDeviceSynchronize());

  dependent_div_seperated<T, N, M>
      <<<blockCount, blockSize>>>((T)0.32, dA, iters);

  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 1; i++) {
    double t1 = dtime();
    dependent_div_seperated<T, N, M>
        <<<blockCount, blockSize>>>((T)0.32, dA, iters);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }
  cudaFree(dA);

  double rcpThru = time.value() * 1380 * 1e6 / N / iters / warpCount;
  cout << setprecision(1) << fixed << typeid(T).name() << " " << setw(5) << N
       << " " << warpCount << " " << setw(5) << M << " "
       << " " << setw(5) << time.value() * 100 << " " << setw(5)
       << time.spread() * 100 << "%   " << setw(5) << setprecision(2) << rcpThru
       << "\n";
  return rcpThru;
}

template <typename T>
map<pair<int, int>, double> measureTabular(int maxWarpCount) {

  map<pair<int, int>, double> results;
  for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {
    results[{warpCount, 1}] = measure<T, 128, 1>(warpCount);
    results[{warpCount, 2}] = measure<T, 128, 2>(warpCount);
    results[{warpCount, 3}] = measure<T, 128, 3>(warpCount);
    results[{warpCount, 4}] = measure<T, 128, 4>(warpCount);
    results[{warpCount, 5}] = measure<T, 128, 5>(warpCount);
    results[{warpCount, 6}] = measure<T, 128, 6>(warpCount);
    results[{warpCount, 7}] = measure<T, 128, 7>(warpCount);
    results[{warpCount, 8}] = measure<T, 128, 8>(warpCount);
    cout << "\n";
  }

  for (int warpCount = 1; warpCount <= maxWarpCount; warpCount *= 2) {
    for (int streams = 1; streams <= 8; streams++) {
      cout << setw(6) << setprecision(2) << results[{warpCount, streams}]
           << " ";
    }
    cout << "\n";
  }

  return results;
}

int main(int argc, char **argv) { measureTabular<double>(32); }
