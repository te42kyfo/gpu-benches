#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>
#include <nvml.h>

using namespace std;

const int max_buffer_size = 512 * 1024 * 1024 + 2;
double *dA, *dB, *dC, *dD;

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 const double *__restrict__ C,
                                 const double *__restrict__ D, const size_t N);

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = 0.1;
  }
}

template <typename T, int unroll>
__global__ void sum_kernel(T *A, const T *__restrict__ B,
                           const T *__restrict__ C, const T *__restrict__ D,
                           const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
   double sum = 0.0;

  for (int64_t i = tidx; i < N;
       i += blockDim.x * gridDim.x * unroll) {
#pragma unroll
      for (int u = 0; u < unroll; u++) {
          sum += B[i + blockDim.x * gridDim.x * u];
      }
  }

  // for (int64_t i = tidx + N - unroll * blockDim.x * gridDim.x; i < N && i >= 0;
  //     i += blockDim.x * gridDim.x) {
  //  sum += B[i];
  // }

  if (tidx == 123123) {
    A[tidx] = sum;
  }
}

template <typename T>
__global__ void dot_kernel(T *A, const T *__restrict__ B,
                           const T *__restrict__ C, const T *__restrict__ D,
                           const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  double sum = 0.0;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    sum += B[i] * C[i];
  }

  if (tidx == 123123) {
    A[tidx] = sum;
  }
}

template <typename T>
__global__ void tdot_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  double sum = 0.0;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    sum += B[i] * C[i] * D[i];
  }

  if (tidx == 123123) {
    A[tidx] = sum;
  }
}
 template <typename T>
__global__ void scale_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = 0.2 * B[i];
  }
}

 template <typename T>
__global__ void scale_kernel_no_GSL(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if( tidx >= N ) return;
  A[tidx] = 0.2 * B[tidx];
}

template <typename T>
__global__ void triad_kernel(T *A, const T *__restrict__ B,
                             const T *__restrict__ C, const T *__restrict__ D,
                             const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] + 0.2 * C[i];
  }
}

template <typename T>
__global__ void sch_triad_kernel(T *A, const T *__restrict__ B,
                                 const T *__restrict__ C,
                                 const T *__restrict__ D, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] + C[i] * D[i];
  }
}

void measureFunc(kernel_ptr_type func, int stream_count, int block_count,
                 int block_size) {

  MeasurementSeries time;
  MeasurementSeries power;

  size_t buffer_size = min(8192, max_buffer_size / (block_count * block_size * 16)) * block_count * block_size * 16;



  func<<<block_count, block_size>>>(dA, dB, dC, dD, buffer_size);

  nvmlDevice_t device;
  int deviceId;
  cudaGetDevice(&deviceId);
  nvmlDeviceGetHandleByIndex(deviceId, &device);

  for (int iter = 0; iter < 9; iter++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());
    func<<<block_count, block_size>>>(dA, dB, dC, dD, buffer_size);
    func<<<block_count, block_size>>>(dA, dB, dC, dD, buffer_size);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1) / 2);
  }

  // for (int iter = 0; iter < 100; iter++) {
  //   func<<<block_count, block_size>>>(dA, dB, dC, dD, buffer_size);
  //   func<<<block_count, block_size>>>(dA, dB, dC, dD, buffer_size);
  //   unsigned int p;
  //   nvmlDeviceGetPowerUsage(device, &p);
  //   power.add(p);
  //   GPU_ERROR(cudaDeviceSynchronize());
  // }

  cout << fixed << setprecision(0)
       << setw(6)
       //<< time.value() * 1000 << " "
       //<< setw(5) << time.spread() * 100
       //<< "   " << setw(5) << power.median() / 1000
       << " " << setw(5)
       << stream_count * buffer_size * sizeof(double) / time.median() * 1e-9;
  cout.flush();
}

void measureKernels(vector<pair<kernel_ptr_type, int>> kernels, int block_count,
                    int block_size, int max_blocks) {
  cout << setw(9) << block_count << "   " << setw(10)
       << block_size * block_count << "  " << setw(7) << setprecision(1)
       << (double)block_count / max_blocks * 100.0 << "  |  GB/s: ";

  for (auto kernel : kernels) {
    measureFunc(kernel.first, kernel.second, block_count, block_size);
  }

  cout << "\n";
}

int main(int argc, char **argv) {
  nvmlInit();
  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));


  init_kernel<<<256, 400>>>(dB, dB, dB, dB, max_buffer_size);
    init_kernel<<<256, 400>>>(dA, dA, dA, dA, max_buffer_size);
  init_kernel<<<256, 400>>>(dC, dC, dC, dC, max_buffer_size);
  init_kernel<<<256, 400>>>(dD, dD, dD, dD, max_buffer_size);
  GPU_ERROR(cudaDeviceSynchronize());

  vector<pair<kernel_ptr_type, int>> kernels = {
      {init_kernel<double>, 1},      {sum_kernel<double, 1>, 1},
       {sum_kernel<double, 2>, 1},   {sum_kernel<double, 4>, 1},
       {sum_kernel<double, 8>, 1},   {sum_kernel<double, 16>, 1},
      {dot_kernel<double>, 2},      {tdot_kernel<double>, 3},
      {scale_kernel<double>, 2},
      {triad_kernel<double>, 3}, {sch_triad_kernel<double>, 4}};

  const int block_size = 256;
  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, kernels[0].first, block_size, 0));

  int max_blocks = maxActiveBlocks * smCount;

  cout << "    blocks     threads     %occ  |               init       sum1    "
          "   sum2  "
          "     sum4       sum8      sum16        dot       tdot      scale "
          "     triad  sch_triad\n";

   for (int block_size = 32; block_size <= 512; block_size *= 2) {
    measureKernels(kernels, 1, block_size, max_blocks);
  }

  for (int i = 2; i < smCount; i*=2) {
    measureKernels(kernels, i, block_size, max_blocks);
  }

  for (int i = smCount; i <= smCount * maxActiveBlocks; i += smCount) {
    measureKernels(kernels, i, block_size, max_blocks);
  }


   GPU_ERROR(cudaFree(dA));
   GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
   GPU_ERROR(cudaFree(dD));
}
