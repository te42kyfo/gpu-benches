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

bool useCudaGraph = false;

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 int size, bool secretlyFalse);

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

template <typename T>
__global__ void scale(T *__restrict__ A, const T *__restrict__ B,
                      const int size, bool secretlyFalse) {

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= size)
    return;

  A[tidx] = B[tidx] * 0.25;
}

void measureFunc(kernel_ptr_type func, int size, dim3 blockSize) {
  MeasurementSeries time;

  dim3 grid = size / blockSize.x + 1;

  func<<<grid, blockSize>>>(dA, dB, size, false);

  int iters = 10000;

  if (useCudaGraph) {

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStream_t stream;
    GPU_ERROR(cudaStreamCreate(&stream));

    GPU_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < iters; i += 2) {
      func<<<grid, blockSize, 0, stream>>>(dA, dB, size, false);
      func<<<grid, blockSize, 0, stream>>>(dB, dA, size, false);
    }
    GPU_ERROR(cudaStreamEndCapture(stream, &graph));
    GPU_ERROR(cudaGraphInstantiate(&instance, graph, NULL, NULL, 0));

    for (int iter = 0; iter < 3; iter++) {
      GPU_ERROR(cudaStreamSynchronize(stream));
      double t1 = dtime();
      GPU_ERROR(cudaGraphLaunch(instance, stream));
      GPU_ERROR(cudaStreamSynchronize(stream));
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
    GPU_ERROR(cudaStreamDestroy(stream));
    GPU_ERROR(cudaGraphDestroy(graph));
    GPU_ERROR(cudaGraphExecDestroy(instance));
  } else {

    func<<<grid, blockSize, 0>>>(dA, dB, size, false);
    for (int iter = 0; iter < 3; iter++) {
      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();
      for (int i = 0; i < iters; i += 2) {
        func<<<grid, blockSize, 0>>>(dA, dB, size, false);
        func<<<grid, blockSize, 0>>>(dB, dA, size, false);
      }
      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
  }
  cout << fixed << setprecision(0)
       << setw(6)
       //<< time.value() * 1000 << " "
       //<< setw(5) << time.spread() * 100
       //<< "   " << setw(5) << power.median() / 1000
       //<< time.median() * 1e6 << "mus "
       << setw(5) << size * 2 * sizeof(double) / time.median() * 1e-9 << "  ";
  ;
  cout.flush();
}

int main(int argc, char **argv) {
  if (argc > 1 && string(argv[1]) == "-graph") {
    useCudaGraph = true;
  }

  GPU_ERROR(cudaMalloc(&dA, max_buffer_count * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_count * sizeof(double)));

  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dA, max_buffer_count);
  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dB, max_buffer_count);

  GPU_ERROR(cudaDeviceSynchronize());

  for (int d = 4 * 1024; d < 16 * 16 * 1024 * 1024;
       d += std::max((int)1, (int)(d * 0.08))) {

    std::cout << d << "  " << d * sizeof(double) * 2 / 1024 << "kB  ";

    for (int xblock = 16; xblock <= 1024; xblock *= 2) {
      measureFunc(scale<double>, d, dim3(xblock, 1, 1));
    }
    std::cout << "\n";
    std::cout.flush();
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
