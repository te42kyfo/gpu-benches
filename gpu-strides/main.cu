#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../gpu-clock.cuh"
#include <array>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#ifdef __NVCC__
#include <nvml.h>
#endif
#ifdef __HIP__
#include <rocm_smi/rocm_smi.h>
#endif

using namespace std;

using dtype = double;
const int max_buffer_size = 32 * 1024 * 1024;
dtype *dA, *dB, *dC, *dD;
using kernel_ptr_type = void (*)(dtype *A, dtype *B, int zero, int one);
unsigned int gpu_clock = 0;

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = T(0.1);
  }
}

template <typename T, int STRIDE, int ITERS>
__global__ void rake_kernel(T *A, T *B, int zero, int one) {
  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % 64;

  T sum = T(0.0);
  const int N = 1000;
#pragma unroll 1

  for (int n = 0; n < N; n++) {
    int ptr = tidx * STRIDE;
    for (int i = 0; i < ITERS; i++) {
      ptr += zero;
      for (int s = 0; s < min(7, STRIDE); s++) {
        sum += A[ptr + s * one] * B[ptr + s * one];
      }
    }
    if (sum == T(123.0)) {
      B[tidx] = T(tidx);
    }
  }
}

template <typename T, int XBLOCK, int PITCH>
__global__ void block_kernel(T *A, T *B, int zero, int one) {
  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % XBLOCK;
  int tidy = (threadIdx.x + blockIdx.x * blockDim.x) / XBLOCK;

  T sum = T(0.0);
  const int N = 1000;

#pragma unroll 1
  for (int n = 0; n < N; n++) {
    for (int i = 0; i < 8; i++) {
      sum +=
          A[tidy * PITCH + tidx + i * zero] * B[tidy * PITCH + tidy + i * zero];
    }

    if (sum == T(123.0)) {
      B[tidx] = T(tidx);
    }
  }
}

void measureFunc(kernel_ptr_type func, int stream_count) {

  MeasurementSeries time;

  int block_count = 1;
  for (int block_size = 64; block_size <= 1024; block_size += 64) {
    func<<<block_count, block_size>>>(dA, dB, 0, 1);

    for (int iter = 0; iter < 21; iter++) {
      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();
      GPU_ERROR(cudaDeviceSynchronize());
      func<<<block_count, block_size>>>(dA + iter, dB + iter, 0, 1);
      func<<<block_count, block_size>>>(dA + iter, dB + iter, 0, 1);
      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add(2 * stream_count * block_size * 1000 * sizeof(dtype) /
               ((t2 - t1) / 2) * 1e-9 / (gpu_clock / 1000.0));
    }
  }

  cout << fixed << setprecision(1) << " " << setw(6) << time.maxValue();
  cout.flush();
}

template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

int main(int argc, char **argv) {

  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));

  init_kernel<<<256, 400>>>(dA, dA, max_buffer_size);
  init_kernel<<<256, 400>>>(dB, dB, max_buffer_size);
  init_kernel<<<256, 400>>>(dC, dC, max_buffer_size);
  init_kernel<<<256, 400>>>(dD, dD, max_buffer_size);
  GPU_ERROR(cudaDeviceSynchronize());

  gpu_clock = getGPUClock();

  std::cout << "\n B/cycle/SM\n";
  std::cout << "pitch    1x64   2x32   4x16    8x8   16x4   32x2   64x1\n";

  constexpr_for<1020, 1026, 1>([](auto i) {
    std::cout << setw(5) << i << " ";
    measureFunc(block_kernel<dtype, 1, i>, 8);
    measureFunc(block_kernel<dtype, 2, i>, 8);
    measureFunc(block_kernel<dtype, 4, i>, 8);
    measureFunc(block_kernel<dtype, 8, i>, 8);
    measureFunc(block_kernel<dtype, 16, i>, 8);
    measureFunc(block_kernel<dtype, 32, i>, 8);
    measureFunc(block_kernel<dtype, 64, i>, 8);
    std::cout << "\n";
  });
  std::cout << "\n Strides: 1-128, B/cycle/SM\n";
  std::cout << "      1      2      3      4      5      6      7      8      9     10     11     12     13     14     15     16\n";

  constexpr_for<1, 128, 1>([](auto i) {
    const int N = std::max(1, 8 / i);
    measureFunc(rake_kernel<dtype, i, N>, std::min(7, (int)i) * N);
    if (i % 16 == 0)
      cout << "\n";
  });

  std::cout << "\n";
  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
}
