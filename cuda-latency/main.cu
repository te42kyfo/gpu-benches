#include "../dtime.hpp"
#include "../gpu-error.h"
#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <nvml.h>
#include <omp.h>
#include <random>
#include <sys/time.h>

using namespace std;

template <typename T>
__global__ void pchase(T *const __restrict__ buf, T *dummy_buf, int iters) {

  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  T idx = 0;

  int iter = 0;
  while (iter < iters) {
    idx = buf[idx];
    if (idx == 0)
      iter++;
  }

  if (tidx > 12313) {
    dummy_buf[0] = idx;
  }
}

int main(int argc, char **argv) {

  nvmlInit();
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(0, &device);
  unsigned int clock = 0;

  typedef uint64_t dtype;

  const int cl_size = 2;
  const int skip_factor = 8;

  for (size_t LEN = 2; LEN < (1 << 28); LEN *= 2) {

    const int iters = max((int64_t)1, ((int64_t)1 << 16) / LEN);
    vector<dtype> order(LEN);
    dtype *buf = NULL;
    dtype *dummy_buf = NULL;

    GPU_ERROR(
        cudaMallocManaged(&buf, skip_factor * cl_size * LEN * sizeof(dtype)));
    GPU_ERROR(cudaMallocManaged(&dummy_buf, sizeof(dtype)));
    for (size_t i = 0; i < LEN; i++) {
      order[i] = i + 1;
    }
    order[LEN - 1] = 0;

    std::random_device rd;
    std::mt19937 g(rd());
    shuffle(begin(order), end(order) - 1, g);

    for (int cl_lane = 0; cl_lane < cl_size; cl_lane++) {
      dtype idx = 0;
      for (size_t i = 0; i < LEN; i++) {

        buf[(idx * cl_size + cl_lane) * skip_factor] =
            skip_factor *
            (order[i] * cl_size + cl_lane + (order[i] == 0 ? 1 : 0));
        idx = order[i];
      }
    }
    buf[skip_factor * (order[LEN - 2] * cl_size + cl_size - 1)] = 0;

    pchase<dtype><<<1, 32>>>(buf, dummy_buf, iters);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
    pchase<dtype><<<1, 32>>>(buf, dummy_buf, iters);
    cudaDeviceSynchronize();
    double start = dtime();
    pchase<dtype><<<1, 32>>>(buf, dummy_buf, iters);
    cudaDeviceSynchronize();
    double end = dtime();

    GPU_ERROR(cudaGetLastError());

    double dt = end - start;
    cout << setw(5) << clock << " " //
         << setw(8) << skip_factor * LEN * cl_size * sizeof(dtype) / 1024
         << " "                                            //
         << fixed                                          //
         << setprecision(1) << setw(5) << dt * 1000 << " " //
         << setw(7) << setprecision(1)
         << (double)dt / iters / (LEN * cl_size) * clock * 1000 * 1000 << "\n";

    GPU_ERROR(cudaFree(buf));
    GPU_ERROR(cudaFree(dummy_buf));
  }
  cout << "\n";
}
