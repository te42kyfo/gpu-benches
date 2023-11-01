#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include <iomanip>
#include <iostream>

using namespace std;

const int64_t max_buffer_count = 512l * 1024 * 1024 + 2;
double *dA, *dB;

#ifdef __NVCC__
const int spoilerSize = 768;
#else
const int spoilerSize = 4 * 1024;
#endif

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 int width, int height, bool secretlyFalse);

template <typename T> __global__ void init_kernel(T *A, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= N)
    return;

  A[tidx] = 0.23;
}

template <typename T, int range>
__global__ void stencilStar2D(T *__restrict__ A, const T *__restrict__ B,
                              const int width, const int height,
                              bool secretlyFalse) {

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  int tidy = threadIdx.y + blockIdx.y * blockDim.y;

  if (tidx >= width - range || tidx < range || tidy >= height - range ||
      tidy < range)
    return;

  A[tidy * width + tidx] = B[tidy * width + tidx];
  for (int i = 0; i < range; i++) {
    A[tidy * width + tidx] +=
        B[tidy * width + tidx + i] + B[tidy * width + tidx - i] +
        B[(tidy + i) * width + tidx] + B[(tidy - i) * width + tidx];
  }

  A[tidy * width + tidx] *= 0.25;
}

void measureFunc(kernel_ptr_type func, int width, int height, dim3 blockSize) {
  /*#ifdef __NVCC__
    if (blocksPerSM == 1) {
      GPU_ERROR(cudaFuncSetAttribute(
          func, cudaFuncAttributePreferredSharedMemoryCarveout, 4));
    } else {
      GPU_ERROR(cudaFuncSetAttribute(
          func, cudaFuncAttributePreferredSharedMemoryCarveout, 9));
    }
  #endif

    int maxActiveBlocks = 0;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks,
                                                            func, blockSize,
  0)); if (maxActiveBlocks != blocksPerSM) cout << "! " << maxActiveBlocks << "
  blocks per SM ";

      */
  MeasurementSeries time;

  dim3 grid = dim3(width / blockSize.x + 1, height / blockSize.y + 1, 1);

  func<<<grid, blockSize>>>(dA, dB, width, height, false);

  for (int iter = 0; iter < 3; iter++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());
    for (int i = 0; i < 1000; i++) {
      func<<<grid, blockSize>>>(dA, dB, width, height, false);
      swap(dA, dB);
    }
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1) / 1000);
  }

  cout << fixed << setprecision(0)
       << setw(6)
       //<< time.value() * 1000 << " "
       //<< setw(5) << time.spread() * 100
       //<< "   " << setw(5) << power.median() / 1000
       //<< time.median() * 1e6 << "mus "
       << setw(5)
       << (width - 2) * (height - 2) * 2 * sizeof(double) / time.median() * 1e-9
       << "  ";
  ;
  cout.flush();
}

int main(int argc, char **argv) {
  GPU_ERROR(cudaMalloc(&dA, max_buffer_count * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_count * sizeof(double)));

  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dA, max_buffer_count);
  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dB, max_buffer_count);

  GPU_ERROR(cudaDeviceSynchronize());

  for (int d = 64; d < 16 * 1024; d += std::max((int)1, (int)(d * 0.03))) {
    std::cout << d << "  ";
    for (int xblock = 4; xblock <= 1024; xblock *= 2) {
      measureFunc(stencilStar2D<double, 0>, d, d,
                  dim3(xblock, 1024 / xblock, 1));
    }
    std::cout << "\n";
    std::cout.flush();
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
