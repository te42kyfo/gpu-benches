#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../gpu-metrics.hpp"
#include <iomanip>
#include <iostream>

using namespace std;

using dtype=double;

dtype *dA, *dB;

__global__ void initKernel(dtype *A, size_t N) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = (dtype) 1.1;
  }
}

template <int N, int iters, int BLOCKSIZE>
__global__ void sumKernel(dtype * __restrict__ A, const dtype * __restrict__ B, int zero) {
  dtype localSum = (dtype)0;

  B += threadIdx.x;
//#pragma unroll(4)
  for (int iter = 0; iter < iters; iter++) {
      B += zero;
      #pragma unroll N/BLOCKSIZE >= 64 ? 32 : N/BLOCKSIZE
      for (int i = 0; i < N; i += BLOCKSIZE) {
          localSum += B[i];
      }
    localSum *= (dtype) 1.3;
  }
  if (localSum == (dtype) 1233)
    A[threadIdx.x] += localSum;
}

template <int N, int iters, int blockSize> double callKernel(int blockCount) {
  sumKernel<N, iters, blockSize><<<blockCount, blockSize>>>(dA, dB, 0);
  return 0.0;
}

template <int N> void measure() {
  const int iters = 100000000 / N + 2;

  const int blockSize = 256;

  cudaDeviceProp prop;
  int deviceId;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sumKernel<N, iters, blockSize>, blockSize, 0));

  int blockCount = smCount * 1; // maxActiveBlocks;

  MeasurementSeries time;
  MeasurementSeries dram_read;
  MeasurementSeries dram_write;
  MeasurementSeries L2_read;
  MeasurementSeries L2_write;

  GPU_ERROR(cudaDeviceSynchronize());


  for (int i = 0; i < 15; i++) {
    const size_t bufferCount = N;// + i * 1282;
    GPU_ERROR(cudaMalloc(&dA, bufferCount * sizeof(dtype)));
    initKernel<<<52, 256>>>(dA, bufferCount);
    GPU_ERROR(cudaMalloc(&dB, bufferCount* sizeof(dtype)));
    initKernel<<<52, 256>>>(dB, bufferCount);
    GPU_ERROR(cudaDeviceSynchronize());

    double t1 = dtime();
    callKernel<N, iters, blockSize>(blockCount);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1));

    measureBandwidthStart();
    callKernel<N, iters, blockSize>(blockCount);
    auto metrics = measureMetricStop();

    dram_read.add(metrics[0]);
    dram_write.add(metrics[1]);
    L2_read.add(metrics[2]*32);
    L2_write.add(metrics[3]*32);

    GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
  }
  double blockDV = N * sizeof(dtype);

  double bw = blockDV * blockCount * iters / time.minValue() / 1.0e9;
  cout << fixed << setprecision(0) << setw(10) << blockDV / 1024 << " kB" //
       << setprecision(0) << setw(10) << time.value() * 1000.0 << "ms"    //
       << setprecision(1) << setw(10) << time.spread() * 100 << "%"       //
       << setw(10) << bw << " GB/s"                                    //
       << setprecision(0) << setw(10) << dram_read.value() / time.minValue() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(10) << dram_write.value() / time.minValue() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(10) << L2_read.value() / time.minValue() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(10) << L2_write.value() / time.minValue() / 1.0e9 << " GB/s " << endl;   //
}

size_t constexpr expSeries(size_t N) {
    size_t val = 32*512;
    for( size_t i  = 0; i < N; i++) {
        val *= 1.17;
    }
    return (val / 512) * 512;
}

int main(int argc, char **argv) {
  cout << setw(13) << "data set" //
       << setw(12) << "exec time" //
       << setw(11) << "spread"    //
       << setw(15) << "Eff. bw" //
       << setw(16) << "DRAM read" //
       << setw(16) << "DRAM write" //
       << setw(16) << "L2 read" //
       << setw(16) << "L2 store\n";

  initMeasureMetric();

  measure<256>();
  measure<512>();
  measure<3*256>();
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
  measure<expSeries(20)>();
  measure<expSeries(21)>();
  measure<expSeries(22)>();
  measure<expSeries(23)>();
  measure<expSeries(24)>();
  measure<expSeries(25)>();
  measure<expSeries(26)>();
  measure<expSeries(27)>();
  measure<expSeries(28)>();
  measure<expSeries(29)>();
  measure<expSeries(30)>();
  measure<expSeries(31)>();
  measure<expSeries(32)>();
  measure<expSeries(33)>();
  measure<expSeries(34)>();
  measure<expSeries(35)>();
  measure<expSeries(36)>();
  measure<expSeries(37)>();
  measure<expSeries(38)>();
  measure<expSeries(39)>();
  measure<expSeries(40)>();
}
