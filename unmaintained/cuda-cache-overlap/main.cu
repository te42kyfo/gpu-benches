#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>
#include <nvml.h>

using namespace std;

double *dA, *dC;

__global__ void initKernel(double *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }
}

template <int N, int ITDV, int BLOCKSIZE>
__global__ __launch_bounds__(1024, 1) void l1kernel(double *A, double *C,
                                                    int otdv,
                                                    int blockMultiplicity,
                                                    int iters, int zeroFactor) {
  double sum = 0.0;
  int blockStart = blockIdx.x / blockMultiplicity * BLOCKSIZE * ITDV * otdv;

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
#pragma unroll(1)
    for (int ot = 0; ot < otdv; ot++) {
      int iteratedBlockStart = blockStart + ot * ITDV * BLOCKSIZE;

      for (int it = 0; it < ITDV; it++) {
        for (int n = 0; n < N; n++) {
          sum += A[iteratedBlockStart + it * BLOCKSIZE + threadIdx.x +
                   n * zeroFactor];
        }
      }
      __syncthreads();
    }
  }

  if (sum == 1.23 && threadIdx.x == 123123) {
    C[threadIdx.x] = sum;
  }
}

template <int N, int ITDV, int BLOCKSIZE>
__global__ __launch_bounds__(1024, 1) void l2kernel(double *A, double *C,
                                                    int otdv,
                                                    int blockMultiplicity,
                                                    int iters, int zeroFactor) {
  double sum = 0.0;
  int blockStart = blockIdx.x / blockMultiplicity * BLOCKSIZE * ITDV * otdv;

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
#pragma unroll(1)
    for (int ot = 0; ot < otdv; ot++) {
      int iteratedBlockStart = blockStart + ot * ITDV * BLOCKSIZE;

#pragma unroll(1)
      for (int n = 0; n < N; n++) {
        for (int it = 0; it < ITDV; it++) {
          sum += A[iteratedBlockStart + it * BLOCKSIZE + threadIdx.x +
                   n * zeroFactor];
        }
      }
      __syncthreads();
    }
  }

  if (sum == 1.23 && threadIdx.x == 123123) {
    C[threadIdx.x] = sum;
  }
}

template <typename T, int N, int ITDV, int BLOCKSIZE>
double callKernel(int otdv, int blockCount, int blockMultiplicity, int iters,
                    bool l1 = false) {
  if (l1)
    l1kernel<N, ITDV, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(
        (T *)dA, (T *)dC, otdv, blockMultiplicity, iters, 0);
  else
    l2kernel<N, ITDV, BLOCKSIZE><<<blockCount, BLOCKSIZE>>>(
        (T *)dA, (T *)dC, otdv, blockMultiplicity, iters, 0);
  return 0.0;
}

auto cachePolicy = cudaFuncCachePreferNone;

template <int N, int ITDV>
void measure(int otdv, int blockCount, int blockMultiplicity, bool l1 = false) {
  const int BLOCKSIZE = 1024;
  const int iters = max(1, 128 * 1024 / ITDV / otdv / N);

  MeasurementSeries time;
  MeasurementSeries dramReadBW;
  MeasurementSeries L2ReadBW;
  MeasurementSeries texReadBW;

  GPU_ERROR(cudaFuncSetCacheConfig(l1kernel<N, ITDV, BLOCKSIZE>, cachePolicy));
  GPU_ERROR(cudaFuncSetCacheConfig(l2kernel<N, ITDV, BLOCKSIZE>, cachePolicy));
  for (int n = 0; n < 5; n++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    callKernel<double, N, ITDV, BLOCKSIZE>(otdv, blockCount, blockMultiplicity,
                                           iters, l1);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }
  GPU_ERROR(cudaGetLastError());

  std::function<double()> measureKernelFunction =
      std::bind(callKernel<double, N, ITDV, BLOCKSIZE>, otdv, blockCount,
                blockMultiplicity, iters, l1);

  for (int i = 0; i < 5; i++) {
    dramReadBW.add(
        measureMetric(measureKernelFunction, "dram_read_throughput") / 1e9);
    L2ReadBW.add(measureMetric(measureKernelFunction, "l2_read_throughput") /
                 1e9);
    texReadBW.add(measureMetric(measureKernelFunction, "tex_cache_throughput") /
                  1e9);
  }

  cout << fixed << setprecision(1)                                   //
       << setw(3) << N << " "                                        //
       << setw(3) << ITDV * sizeof(double) * BLOCKSIZE / 1024 << " " //
       << setw(3)
       << ITDV * sizeof(double) * BLOCKSIZE / 1024 * blockCount /
              blockMultiplicity
       << " " //
       << setw(3)
       << otdv * ITDV * sizeof(double) * BLOCKSIZE / 1024 * blockCount /
              blockMultiplicity
       << " "                                    //
       << setw(5) << time.median() * 1000 << " " //
       << setw(5)
       << sizeof(double) * otdv * ITDV * BLOCKSIZE * blockCount * N * iters /
              time.median() / 1e9
       << "  -  " //
       << setw(5) << texReadBW.median() << "  " << setw(5) << L2ReadBW.median()
       << "  " << setw(5) << dramReadBW.median() << " \n";
}

int main(int argc, char **argv) {
  // nvmlInit();
  measureMetricInit();

  GPU_ERROR(cudaMalloc(&dA, 1024 * 1024 * 256 * sizeof(double)));
  initKernel<<<52, 256>>>(dA, 1024 * 1024 * 256);
  GPU_ERROR(cudaMalloc(&dC, 1024 * 1024 * 256 * sizeof(double)));
  initKernel<<<52, 256>>>(dC, 1024 * 1024 * 256);
  GPU_ERROR(cudaGetLastError());

  int blockMultiplicity = 1;
  int blockCount = 1;
  cachePolicy = cudaFuncCachePreferL1;

  const int ITDV_L1 = 12;
  int otdv_l1 = 1;
  measure<1, ITDV_L1>(otdv_l1, blockCount, blockMultiplicity, true);
  measure<2, ITDV_L1>(otdv_l1, blockCount, blockMultiplicity, true);
  measure<4, ITDV_L1>(otdv_l1, blockCount, blockMultiplicity, true);
  measure<8, ITDV_L1>(otdv_l1, blockCount, blockMultiplicity, true);
  measure<16, ITDV_L1>(otdv_l1, blockCount, blockMultiplicity, true);
  measure<32, ITDV_L1>(otdv_l1, blockCount, blockMultiplicity, true);

  cout << "\n";

  int otdv_l2 = 32;
  measure<1, ITDV_L1>(otdv_l2, blockCount, blockMultiplicity, true);
  measure<2, ITDV_L1>(otdv_l2, blockCount, blockMultiplicity, true);
  measure<4, ITDV_L1>(otdv_l2, blockCount, blockMultiplicity, true);
  measure<8, ITDV_L1>(otdv_l2, blockCount, blockMultiplicity, true);
  measure<16, ITDV_L1>(otdv_l2, blockCount, blockMultiplicity, true);
  measure<32, ITDV_L1>(otdv_l2, blockCount, blockMultiplicity, true);

  cout << "\n";

  int otdv_mem = 1024;
  measure<1, ITDV_L1>(otdv_mem, blockCount, blockMultiplicity, true);
  measure<2, ITDV_L1>(otdv_mem, blockCount, blockMultiplicity, true);
  measure<4, ITDV_L1>(otdv_mem, blockCount, blockMultiplicity, true);
  measure<8, ITDV_L1>(otdv_mem, blockCount, blockMultiplicity, true);
  measure<16, ITDV_L1>(otdv_mem, blockCount, blockMultiplicity, true);
  measure<32, ITDV_L1>(otdv_mem, blockCount, blockMultiplicity, true);

  cout << "\n";

  cachePolicy = cudaFuncCachePreferShared;
  blockCount = 80;
  const int ITDV_L2 = 6;
  blockMultiplicity = 1;
  measure<1, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<2, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<4, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<8, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<16, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<32, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<64, ITDV_L2>(4, blockCount, blockMultiplicity);
  measure<128, ITDV_L2>(4, blockCount, blockMultiplicity);

  cout << "\n";
  measure<1, ITDV_L2>(32, blockCount, blockMultiplicity);
  measure<2, ITDV_L2>(32, blockCount, blockMultiplicity);
  measure<4, ITDV_L2>(32, blockCount, blockMultiplicity);
  measure<8, ITDV_L2>(32, blockCount, blockMultiplicity);
  measure<16, ITDV_L2>(32, blockCount, blockMultiplicity);
  measure<64, ITDV_L2>(32, blockCount, blockMultiplicity);
  measure<128, ITDV_L2>(32, blockCount, blockMultiplicity);
}
