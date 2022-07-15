#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../measure_metric/measureMetricPW.hpp"
#include <iomanip>
#include <iostream>

using namespace std;

double *dA, *dB;

__global__ void initKernel(double *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = 1.1;
  }
}

template <int N, int BLOCKSIZE>
__global__ void sumKernel(double *__restrict__ A, const double *__restrict__ B,
                          int blockRun) {
  double localSum = 0;

  for (int i = 0; i < N; i++) {
    int idx = BLOCKSIZE * blockRun * i + (blockIdx.x % blockRun) * BLOCKSIZE +
              threadIdx.x;
    localSum += B[idx];
    //A[idx] = 1.23 * B[idx];
  }
  localSum *= 1.3;
  if (threadIdx.x > 1233 || localSum == 23.12)
    A[threadIdx.x] += localSum;
}



template <int N, int blockSize> double callKernel(int blockCount, int blockRun) {
  sumKernel<N, blockSize><<<blockCount, blockSize>>>(dA, dB, blockRun);
  return 0.0;
}

template <int N> void measure() {

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

  int blockRun = smCount * maxActiveBlocks;
  int blockCount = blockRun * 10000;

  GPU_ERROR(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(cudaDeviceSynchronize());
  for (int i = 0; i < 9; i++) {
    const size_t bufferCount = blockRun * blockSize * N + i * 128;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(double)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount * sizeof(double)));
    initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(cudaDeviceSynchronize());

    double t1 = dtime();
    callKernel<N, blockSize>(blockCount, blockRun);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);

    measureMetricStart({"dram__bytes_read.sum", "dram__bytes_write.sum",
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_write.sum"});
    callKernel<N, blockSize>(blockCount, blockRun);
    auto metrics = measureMetricStop();

    dram_read.add(metrics[0]);
    dram_write.add(metrics[1]);
    L2_read.add(metrics[2] * 32);
    L2_write.add(metrics[3] * 32);

    cudaFree(dA);
    cudaFree(dB);
  }

  double blockDV = N * blockSize * sizeof(double);

  double bw = blockDV * blockCount  / time.median() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << fixed << setprecision(0) << setw(10) << blockDV * blockRun / 1024
       << " kB"                                                        //
       << setprecision(0) << setw(10) << time.median() * 1000.0 << "ms" //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"    //
       << setw(10) << bw << " GB/s   "                                 //
       << setprecision(0) << setw(10)
       << dram_read.median() / time.median() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10)
       << dram_write.median() / time.median() / 1.0e9 << " GB/s " //
       << setprecision(0) << setw(10) << L2_read.median() / time.median() / 1.0e9
       << " GB/s " //
       << setprecision(0) << setw(10) << L2_write.median() / time.median() / 1.0e9
       << " GB/s " << endl; //
}

size_t constexpr expSeries(size_t N) {
  size_t val = 20;
  for (size_t i = 0; i < N; i++) {
    val = val * 1.04 + 1;
  }
  return val;
}

int main(int argc, char **argv) {

  cout << setw(13) << "data set"   //
       << setw(12) << "exec time"  //
       << setw(11) << "spread"     //
       << setw(15) << "Eff. bw\n"; //

  measure<1>();
  measure<2>();
  measure<3>();
  measure<4>();
  measure<5>();
  measure<6>();
  measure<7>();
  measure<8>();
  measure<9>();
  measure<10>();
  measure<11>();
  measure<12>();
  measure<13>();
  measure<14>();
  measure<15>();
  measure<16>();
  measure<17>();
  measure<18>();
  measure<19>();
  measure<expSeries(0)>();
  measure<expSeries(1)>();
  measure<expSeries(2)>();
  measure<expSeries(3)>();
  measure<expSeries(4)>();
  measure<expSeries(5)>();
  measure<expSeries(6)>();
  /*measure<expSeries(7)>();
  measure<expSeries(8)>();
  measure<expSeries(9)>();
  measure<expSeries(10)>();
  measure<expSeries(11)>();
  measure<expSeries(12)>();
  measure<expSeries(13)>();
  measure<expSeries(14)>();
  measure<expSeries(15)>();
  measure<expSeries(16)>();
  measure<expSeries(17)>();
  measure<expSeries(18)>();
  measure<expSeries(19)>();
  measure<expSeries(20)>();*/
}
