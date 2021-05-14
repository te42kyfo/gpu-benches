#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>

using namespace std;

double *dA, *dB, *dC;

__global__ void initKernel(double *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }
}

template <int N, int iters, int BLOCKSIZE>
__global__ void daxpyKernel(double *A, double *B, double *C) {
  double localSum = 0;
#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
    for (int i = 0; i < N; i += BLOCKSIZE) {
      int idx = i + threadIdx.x;
      localSum += B[idx] * C[idx];
    }
    localSum *= 1.3;
    if (threadIdx.x > 1233)
      A[threadIdx.x + blockIdx.x * blockDim.x] = 2.3;
  }
  if (threadIdx.x > 1233)
    A[threadIdx.x] += localSum;
}

template <int N, int iters, int blockSize> double callKernel(int blockCount) {
  daxpyKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, dC);
  return 0.0;
}

template <int N> void measure() {
  const int iters = 100000000 / N + 1;

  const int blockSize = 256;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, daxpyKernel<N, iters, blockSize>, blockSize, 0));

  int blockCount = smCount * 1; // maxActiveBlocks;

  MeasurementSeries time;

  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 15; i++) {
    GPU_ERROR(cudaMalloc(&dA, (N + i * 128) * sizeof(double)));
    initKernel<<<52, 256>>>(dA, N + i * 128);
    GPU_ERROR(cudaMalloc(&dB, (N + i * 128) * sizeof(double)));
    initKernel<<<52, 256>>>(dB, N + i * 128);
    GPU_ERROR(cudaMalloc(&dC, (N + i * 128) * sizeof(double)));
    initKernel<<<52, 256>>>(dC, N + i * 128);
    GPU_ERROR(cudaDeviceSynchronize());

    double t1 = dtime();
    callKernel<N, iters, blockSize>(blockCount);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
  }
  GPU_ERROR(cudaMalloc(&dA, N * sizeof(double)));
  initKernel<<<52, 256>>>(dA, N);
  GPU_ERROR(cudaMalloc(&dB, N * sizeof(double)));
  initKernel<<<52, 256>>>(dB, N);
  GPU_ERROR(cudaMalloc(&dC, N * sizeof(double)));
  initKernel<<<52, 256>>>(dC, N);

  GPU_ERROR(cudaDeviceSynchronize());
  std::function<double()> measureKernelFunction =
      std::bind(callKernel<N, iters , blockSize>, blockCount);

  double dramReadBW =
      measureMetric(measureKernelFunction, "dram_read_throughput") / 1e9;
  double dramWriteBW =
      measureMetric(measureKernelFunction, "dram_write_throughput") / 1e9;

  double L2ReadBW =
      measureMetric(measureKernelFunction, "l2_read_throughput") / 1e9;

  double L2WriteBW =
      measureMetric(measureKernelFunction, "l2_write_throughput") / 1e9;

  double texReadBW =
      measureMetric(measureKernelFunction, "tex_cache_throughput") / 1e9;

  cudaFree(dA);
  cudaFree(dB);
  cudaFree(dC);

  double blockDV = N * sizeof(double) * 2;

  double bw = blockDV * blockCount * iters / time.value() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
       << setw(10) << bw << " GB/s"                                        //
       << setw(10) << dramReadBW << " GB/s"                                //
       << setw(10) << dramWriteBW << " GB/s"                               //
       << setw(10) << L2ReadBW << " GB/s"                                  //
       << setw(10) << L2WriteBW << " GB/s"                                 //
       << setw(10) << texReadBW << " GB/s\n";
}

size_t constexpr expSeries(size_t N) {
    size_t val = 32*512;
    for( size_t i  = 0; i < N; i++) {
        val *= 1.15;
    }
    return (val / 512) * 512;
}

int main(int argc, char **argv) {
  measureMetricInit();

  cout << setw(13) << "data set" //
       << setw(12) << "exec time" //
       << setw(11) << "spread"    //
       << setw(15) << "Eff. bw"   //
       << setw(15) << "DRAM read"  //
       << setw(15) << "DRAM write" //
       << setw(15) << "L2 Read"    //
       << setw(15) << "L2 Write"   //
       << setw(15) << "Tex Read\n";

  measure<1*512>();
  measure<2*512>();
  measure<3*512>();
  measure<4*512>();
  measure<5*512>();
  measure<6*512>();
  measure<7*512>();
  measure<8*512>();
  measure<9*512>();
  measure<10*512>();
  measure<11*512>();
  measure<12*512>();
  measure<13*512>();
  measure<14*512>();
  measure<15*512>();
  measure<16*512>();
  measure<17*512>();
  measure<18*512>();
  measure<19*512>();
  measure<20*512>();
  measure<21*512>();
  measure<22*512>();
  measure<23*512>();
  measure<24*512>();
  measure<25*512>();
  measure<26*512>();
  measure<27*512>();
  measure<28*512>();
  measure<29*512>();
  measure<30*512>();
  measure<31*512>();
  measure<32*512>();

  measure<expSeries(1)>();
  measure<expSeries(2)>();
  measure<expSeries(3)>();
  measure<expSeries(4)>();
  measure<expSeries(5)>();
  measure<expSeries(6)>();
  measure<expSeries(7)>();
  measure<expSeries(8)>();
  measure<expSeries(9)>();
  measure<expSeries(10)>();
  measure<expSeries(11)>();
  measure<expSeries(12)>();
  measure<expSeries(13)>();
  measure<expSeries(14)>();
  measure<expSeries(16)>();
  measure<expSeries(17)>();
  measure<expSeries(18)>();
  measure<expSeries(19)>();
}
