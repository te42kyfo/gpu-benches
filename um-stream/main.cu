#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include <iomanip>
#include <iostream>
using namespace std;

__global__ void scale(double *A, double *B, size_t N) {
  size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] * 1.3;
  }
}

__global__ void triad(double *A, double *B, double *C, size_t N) {
  size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] + C[i] * 1.3;
  }
}

int main(int argc, char **argv) {
  double *A, *B, *C;

  cout << setw(12) << "buffer size" << setw(10) << "time" << setw(9) << "spread"
       << setw(13) << "bandwidth\n";

  const int blockSize = 256;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                          triad, blockSize, 0));

  int blockCount = smCount * maxActiveBlocks;

  for (size_t N = 1024 * 1024; N < (size_t)1024 * 1024 * 1024 * 16; N *= 2) {
    GPU_ERROR(cudaMallocManaged(&A, N * sizeof(double)));
    GPU_ERROR(cudaMallocManaged(&B, N * sizeof(double)));
    GPU_ERROR(cudaMallocManaged(&C, N * sizeof(double)));

    triad<<<blockCount, blockSize>>>(A, B, C, N);
    //	scale<<<640, 256>>>(A, B, N);
    GPU_ERROR(cudaDeviceSynchronize());

    MeasurementSeries time;
    for (int i = 0; i < 5; i++) {
      double t1 = dtime();
      triad<<<640, 256>>>(A, B, C, N);
      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add(t2 - t1);
    }

    double bw = N * sizeof(double) * 3 / time.value() / 1.0e9;
    cout << fixed << setprecision(1) << setw(9)
         << 3 * N * sizeof(double) / (1 << 20) << " MB" << setw(8)
         << time.value() * 1000 << "ms" << setprecision(1) << setw(8)
         << time.spread() * 100 << "%" << setw(8) << bw << "GB/s\n";

    GPU_ERROR(cudaFree(A));
    GPU_ERROR(cudaFree(B));
    GPU_ERROR(cudaFree(C));
  }
}
