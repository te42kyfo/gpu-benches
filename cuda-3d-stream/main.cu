#include "../MeasurementSeries.hpp"
#include "../measure_metric/measureMetricPW.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include <iomanip>
#include <iostream>
#include <nvml.h>

using namespace std;

const size_t xdim = 2000;
const size_t ydim = 1000;
const size_t zdim = 100;
const size_t buffer_size = (size_t) xdim * ydim * zdim;
double *dA, *dB;

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = 0.1;
  }
}

template <typename T>
__global__ void scale_kernel(T *A, const T *__restrict__ B) {
  __shared__ double spoiler[1024];
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;
  int tidz = threadIdx.z + blockIdx.z * blockDim.z;
  if (tidx >= xdim || tidy >= ydim || tidz >= zdim)
    return;

  if (threadIdx.x > 1243)
    spoiler[threadIdx.x] = B[threadIdx.x];

  size_t idx = tidz * xdim * ydim + tidy * xdim + tidx;
  A[idx] = B[idx] * 1.2;

  if (threadIdx.x > 1243)
    A[idx] = spoiler[idx];
}

void measureFunc(dim3 blockSize) {

  GPU_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
  MeasurementSeries time;

  dim3 grid = dim3((xdim - 1) / blockSize.x + 1, (ydim - 1) / blockSize.y + 1,
                   (zdim - 1) / blockSize.z + 1);

  scale_kernel<<<grid, blockSize>>>(dA, dB);

  nvmlDevice_t device;
  int deviceId;
  cudaGetDevice(&deviceId);
  nvmlDeviceGetHandleByIndex(deviceId, &device);

  for (int iter = 0; iter < 10; iter++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());
    scale_kernel<<<grid, blockSize>>>(dA, dB);
    scale_kernel<<<grid, blockSize>>>(dA, dB);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1) / 2);
  }

  measureBandwidthStart();
  scale_kernel<<<grid, blockSize>>>(dA, dB);
  auto metrics = measureMetricStop();

  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, scale_kernel<double>, blockSize.x*blockSize.y*blockSize.z, 0));


  cout << fixed << setprecision(0) << "(" << setw(4) << blockSize.x << ","
       << setw(4) << blockSize.y << "," << setw(4) << blockSize.z << ")      "
       << maxActiveBlocks << " "
       << setw(2) << " " << setw(5)
       << buffer_size * 2 * sizeof(double) / time.median() * 1e-9 << "  "
       << (maxActiveBlocks*smCount*blockSize.x*blockSize.y*blockSize.z) * time.median() * 1.41e9 / buffer_size << " "

       << setprecision(0) << setw(8) << metrics[0] / time.value() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(8) << metrics[1] / time.value() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(8) << metrics[2]*32 / time.value() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(8) << metrics[3]*32 / time.value() / 1.0e9 << " GB/s " << endl;   //
  cout.flush();
}

int main(int argc, char **argv) {
  nvmlInit();
  GPU_ERROR(cudaMalloc(&dA, buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, buffer_size * sizeof(double)));

  init_kernel<<<256, 400>>>(dB, dB, dB, dB, buffer_size);
  init_kernel<<<256, 400>>>(dA, dA, dA, dA, buffer_size);
  GPU_ERROR(cudaDeviceSynchronize());

  for (int blockDimX : {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
    for (int blockDimY : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}) {
      for (int blockDimZ : {1, 2, 4, 8, 16, 32, 64}) {
          int threadCount = blockDimX * blockDimY * blockDimZ;

        if (threadCount != 256) //threadCount > 1024 || threadCount < 64)
          continue;

        measureFunc(dim3(blockDimX, blockDimY, blockDimZ));
      }
    }
  }

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
