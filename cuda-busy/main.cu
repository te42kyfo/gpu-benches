#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>

using namespace std;

double *dA;

__global__ void kernel(int iters, int N, double *A) {
  double sum = 0.0;

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
#pragma unroll(1)
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
      sum += A[n];
    }
  }

  if (sum == -12.3) {
    A[threadIdx.x] = sum;
  }
}

void measure(int N) {
  int iters = 10000;
  int blockCount = 1;
  int blockSize = 32;

  MeasurementSeries time;
  for (int i = 0; i < 10; i++) {

    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();

    kernel<<<blockCount, blockSize>>>(iters, N, dA);

    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }
  GPU_ERROR(cudaGetLastError());

  double spread = (time.median() - time.minValue()) / time.median() * 100;
  double dt = time.minValue();
  double bw = N * iters * sizeof(double) / dt / 1e9;
  double cyc = dt / (N * iters) * 1.38e9 * 32;

  cout << fixed << setprecision(1);
  cout << setw(8) << dt * 1000 << "   " //
       << setw(8) << spread << "   "    //
       << setw(8) << bw << "   "        //
       << setw(8) << cyc << "\n";
}

int main(int argc, char **argv) {

  size_t maxBufferSize = 1024 * 1024;
  GPU_ERROR(cudaMallocManaged(&dA, sizeof(double) * maxBufferSize));
  for (size_t i = 0; i < maxBufferSize; i++) {
    dA[i] = 1.2;
  }

  measure(8 * 1024);

  GPU_ERROR(cudaFree(dA));
  return 0;
}
