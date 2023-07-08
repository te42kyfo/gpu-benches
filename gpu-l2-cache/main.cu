#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"
#include <iomanip>
#include <iostream>

using namespace std;

using dtype = double;

dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }
}

template <int N, int BLOCKSIZE>
__global__ void sumKernel(dtype *__restrict__ A, const dtype *__restrict__ B,
                          int blockRun) {
  dtype localSum = 0;

  for (int i = 0; i < N; i++) {
    int idx = blockDim.x * blockRun * i + (blockIdx.x % blockRun) * BLOCKSIZE +
              threadIdx.x;
    localSum += B[idx];
    // A[idx] = 1.23 * B[idx];
  }
  localSum *= (dtype)1.3;
  if (threadIdx.x > 1233 || localSum == (dtype)23.12)
    A[threadIdx.x] += localSum;
}

template <int N, int blockSize> dtype callKernel(int blockCount, int blockRun) {
  sumKernel<N, blockSize><<<blockCount, blockSize>>>(dA, dB, blockRun);
  GPU_ERROR(cudaPeekAtLastError());
  return 0.0;
}

template <int N> void measure(int blockRun) {

  const int blockSize = 1024;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, blockSize>, blockSize, 0));

  int blockCount = 100000;

  // GPU_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 9; i++) {
    const size_t bufferCount = blockRun * blockSize * N + i * 128;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(cudaDeviceSynchronize());

    double t1 = dtime();
    callKernel<N, blockSize>(blockCount, blockRun);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);

    // measureMetricsStart({"dram__bytes_read.sum", "dram__bytes_write.sum",
    //                      "lts__t_sectors_srcunit_tex_op_read.sum",
    //                      "lts__t_sectors_srcunit_tex_op_write.sum"});
    //
    // measureMetricsStart({"GL2C_MISS_sum"});

    // callKernel<N, blockSize>(blockCount, blockRun);
    // auto metrics = measureMetricStop();
    // dram_read.add(metrics[0] * 1024);
    //  dram_write.add(metrics[1]);
    //    L2_read.add(metrics[2] * 32);
    //    L2_write.add(metrics[3] * 32);

    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
  }

  double blockDV = N * blockSize * sizeof(dtype);

  double bw = blockDV * blockCount / time.median() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << fixed << setprecision(0) << setw(10) << blockDV * blockRun / 1024
       << " kB"                                                         //
       << setprecision(0) << setw(10) << time.median() * 1000.0 << "ms" //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"     //
       << setw(10) << bw << " GB/s   "                                  //
       << setprecision(0) << setw(6) << dram_read.median() << " GB/s "  //
       << setprecision(0) << setw(6)
       << dram_write.median() / time.median() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(6) << L2_read.median() / time.median() / 1.0e9
       << " GB/s " //
       << setprecision(0) << setw(6)
       << L2_write.median() / time.median() / 1.0e9 << " GB/s " << endl; //
}

size_t constexpr expSeries(size_t N) {
  size_t val = 20;
  for (size_t i = 0; i < N; i++) {
    val = val * 1.04 + 1;
  }
  return val;
}

int main(int argc, char **argv) {
  initMeasureMetric();
  cout << setw(13) << "data set"   //
       << setw(12) << "exec time"  //
       << setw(11) << "spread"     //
       << setw(15) << "Eff. bw\n"; //

  for (int i = 1; i < 100000; i += max(1.0, i * 0.1)) {
    measure<32>(i);
  }
}
