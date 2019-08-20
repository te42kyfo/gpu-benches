#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>

using namespace std;

double *dA, *dB;

using kernel_ptr_type = void (*)(int iters, double *A, const double *B);

template <int N, int UNROLL, bool DOTPRODUCT>
__global__ __launch_bounds__(32, 1) void kernel(int iters, double *A,
                                                double *B) {
  double sum = 0.0;

  double *dA = A + threadIdx.x;
  double *dB = B + threadIdx.x;

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
#pragma unroll(UNROLL)
    for (int n = 0; n < N; n++) {
      if (DOTPRODUCT)
        sum += dA[n * 32] * dB[n * 32];
      else
        sum += dA[n * 32];
    }
  }

  if (sum == -12.3) {
    A[threadIdx.x] = sum;
  }
}

template <int DV, int UNROLL, bool DOTPRODUCT> void measure() {

  int blockCount = 1;
  const int blockSize = 32;
  const int N = DV / blockSize;
  int iters = 100000 / N;

  GPU_ERROR(cudaFuncSetCacheConfig(kernel<N, UNROLL, DOTPRODUCT>,
                                   cudaFuncCachePreferL1));

  MeasurementSeries time;
  for (int i = 0; i < 20; i++) {

    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();

    kernel<N, UNROLL, DOTPRODUCT><<<blockCount, blockSize>>>(iters, dA, dB);

    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }
  GPU_ERROR(cudaGetLastError());

  double spread = (time.median() - time.minValue()) / time.median() * 100;
  double dt = time.minValue();
  double bw = (DOTPRODUCT ? 2 : 1) * DV * iters * sizeof(double) / dt / 1e9;
  double cyc = dt / (DV * iters) * 1.38e9 * 32;

  cout << fixed << setprecision(1);

  cout << setw(3) << UNROLL << "  "     //
       << setw(8) << dt * 1000 << "   " //
       << setw(8) << spread << "   "    //
       << setw(8) << bw << "   "        //
       << setw(8) << cyc << " -- " << setw(8)
       << (20.0 + max((UNROLL-1) * (DOTPRODUCT ? 8 : 4), 30) + UNROLL * 8) / UNROLL << " \n";
}

int main(int argc, char **argv) {

  size_t maxBufferSize = 1024 * 1024;
  GPU_ERROR(cudaMallocManaged(&dA, sizeof(double) * maxBufferSize));
  GPU_ERROR(cudaMallocManaged(&dB, sizeof(double) * maxBufferSize));
  for (size_t i = 0; i < maxBufferSize; i++) {
    dA[i] = 1.2;
    dB[i] = 1.21;
  }

  measure<9 * 512, 1, true>();
  measure<9 * 512, 2, true>();
  measure<9 * 512, 3, true>();
  measure<9 * 512, 4, true>();
  measure<9 * 512, 6, true>();
  measure<9 * 512, 8, true>();
  measure<9 * 512, 9, true>();
  measure<9 * 512, 12, true>();
  measure<9 * 512, 16, true>();
  measure<9 * 512, 18, true>();
  measure<9 * 512, 24, true>();
  measure<9 * 512, 27, true>();
  measure<9 * 512, 32, true>();
  cout << "\n";
  measure<9 * 512, 1, false>();
  measure<9 * 512, 2, false>();
  measure<9 * 512, 3, false>();
  measure<9 * 512, 4, false>();
  measure<9 * 512, 6, false>();
  measure<9 * 512, 8, false>();
  measure<9 * 512, 9, false>();
  measure<9 * 512, 12, false>();
  measure<9 * 512, 16, false>();
  measure<9 * 512, 18, false>();
  measure<9 * 512, 24, false>();
  measure<9 * 512, 27, false>();
  measure<9 * 512, 32, false>();

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  return 0;
}
