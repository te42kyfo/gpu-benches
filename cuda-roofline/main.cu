#include "../dtime.hpp"
#include "../gpu-error.h"
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <omp.h>
#include <sys/time.h>
#include <unistd.h>

using namespace std;

template <typename T> __global__ void initKernel(T *data, size_t data_len) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int idx = tidx; idx < data_len; idx += gridDim.x * blockDim.x) {
    data[idx] = idx;
  }
}

template <typename T, int N, int M, int BLOCKSIZE>
__global__ void testfun(T *const __restrict__ dA, T *const __restrict__ dB,
                        T *dC) {
  T *sA = dA + threadIdx.x + blockIdx.x * BLOCKSIZE * M;
  T *sB = dB + threadIdx.x + blockIdx.x * BLOCKSIZE * M;

  T sum = 0;

#pragma unroll 1
  for (int i = 0; i < M; i+=2) {
    T a = sA[i * BLOCKSIZE];
    T b = sB[i * BLOCKSIZE];
    T v = a - b;
    T a2 = sA[(i+1) * BLOCKSIZE];
    T b2 = sB[(i+1) * BLOCKSIZE];
    T v2 = a2 - b2;
    for (int i = 0; i < N; i++) {
      v = v * a - b;
      v2 = v2 * a2 - b2;
    }
    sum += v + v2;
  }
  if (threadIdx.x == 0)
    dC[blockIdx.x] = sum;
}


template <typename T, int N, int M, int BLOCKSIZE>
__global__ void testfun_max_power(T *const __restrict__ dA, T *const __restrict__ dB,
                        T *dC) {
  T *sA = dA + (threadIdx.x) + (blockIdx.x/2) * BLOCKSIZE * (M/8);
  T *sB = dB + (threadIdx.x) + (blockIdx.x/2) * BLOCKSIZE * (M/8);

  T sum = 0;
#pragma unroll 1
  for (int i = 0; i < M; i +=2) {
    T a = sA[(i / 64) * BLOCKSIZE];
    T b = sB[(i / 64) * BLOCKSIZE];
    T v = a - b;
    T a2 = sA[(i / 64 + 1) * BLOCKSIZE];
    T b2 = sB[(i / 64 + 1) * BLOCKSIZE];
    T v2 = a2 - b2;
    for (int i = 0; i < N; i++) {
      v = v * a - a;
      v2 = v2 * a - a;
    }
    sum += v + v2;
  }
  if (threadIdx.x == 0)
    dC[blockIdx.x] = sum;
}

int main(int argc, char **argv) {
  nvmlInit();

  typedef float dtype;
  const int M = 8000;
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

     nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(deviceId, &device);
    unsigned int power = 0;
    unsigned int clock = 0;
    unsigned int temperature = 0;

    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks, testfun<dtype, N, M, BLOCKSIZE>, BLOCKSIZE, 0));
    int blockCount = prop.multiProcessorCount * numBlocks;

    size_t data_len = (size_t)blockCount * BLOCKSIZE * M;
    dtype *dA = NULL;
    dtype *dB = NULL;
    dtype *dC = NULL;
    size_t iters = 500;

    GPU_ERROR(cudaMalloc(&dA, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dB, data_len * sizeof(dtype)));
    GPU_ERROR(cudaMalloc(&dC, data_len * sizeof(dtype)));
#pragma omp barrier
    initKernel<<<blockCount, 256>>>(dA, data_len);
    initKernel<<<blockCount, 256>>>(dB, data_len);
    initKernel<<<blockCount, 256>>>(dC, data_len);
    cudaDeviceSynchronize();


#pragma omp barrier

    double start = dtime();
    for (size_t iter = 0; iter < iters; iter++) {
      testfun_max_power<dtype, N, M, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(dA, dB, dC);
    }
    usleep(1000000);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
    nvmlDeviceGetPowerUsage(device, &power);
    nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);
    cudaDeviceSynchronize();
    double end = dtime();
    GPU_ERROR(cudaGetLastError());

#pragma omp barrier
#pragma omp for ordered schedule(static, 1)
    for (int i = 0; i < omp_get_num_threads(); i++) {
#pragma omp ordered
      {
        cout << setprecision(3) << fixed << deviceId << " " << blockCount
             << " blocks   " << setw(3) << N << " its      "
             << (2.0 + N * 2.0) / (2.0 * sizeof(dtype)) << " Fl/B      "
             << setprecision(0) << setw(5)
             << iters * 2 * data_len * sizeof(dtype) / (end - start) * 1.0e-9
             << " GB/s    " << setw(6)
             << iters * (2 + N * 2) * data_len / (end - start) * 1.0e-9
             << " GF/s   " << clock << " Mhz   " << power / 1000 << " W   "
             << temperature << "°C\n";
      }
    }
    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
    GPU_ERROR(cudaFree(dC));
  }
  cout << "\n";
}
