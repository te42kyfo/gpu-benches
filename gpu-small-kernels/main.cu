#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include <cooperative_groups.h>
#include <iomanip>
#include <iostream>

namespace cg = cooperative_groups;

using namespace std;

const int64_t max_buffer_count = 512l * 1024 * 1024 + 2;
double *dA, *dB;

#ifdef __NVCC__
const int spoilerSize = 768;
#else
const int spoilerSize = 4 * 1024;
#endif

bool useCudaGraph = false;
bool usePersistentThreadsAtomic = false;
bool usePersistentThreadsGsync = false;

using kernel_ptr_type = void (*)(double *A, const double *__restrict__ B,
                                 int size);

template <typename T> __global__ void init_kernel(T *A, size_t N, T val) {
  size_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
  for (int idx = tidx; idx < N; idx += blockDim.x * gridDim.x) {
    A[idx] = (T)val;
  }
}

template <typename T>
__global__ void scale(T *__restrict__ A, const T *__restrict__ B,
                      const int size) {

  int tidx = threadIdx.x + blockIdx.x * blockDim.x;

  if (tidx >= size)
    return;

  A[tidx] = B[tidx] * 0.25;
}

template <typename T>
__global__ void sync_kernel_gsync(T *__restrict__ A, T *__restrict__ B,
                                  int size, int iters) {

  cg::grid_group g = cg::this_grid();
  int tidx = g.thread_rank();

  for (int iter = 0; iter < iters; iter++) {
    for (int id = tidx; id < size; id += blockDim.x * gridDim.x) {
      A[id] = B[id] * 0.25;
    }

    g.sync();
  }
}
template <typename T>
__global__ void sync_kernel_atomic(volatile int *flags, T *__restrict__ A,
                                   T *__restrict__ B, int size, int iters) {

  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int threadCount = gridDim.x;

  for (int iter = 0; iter < iters; iter++) {
    for (int id = tidx; id < size; id += blockDim.x * gridDim.x) {

      A[id] = B[id] * 0.25;
    }

    __syncthreads();
    __threadfence();
    int old_val;
    if (threadIdx.x == 0) {
      old_val = atomicAdd((int *)&(flags[iter]), 1);
      while (flags[iter] != threadCount)
        ;
    }
    __syncthreads();
  }
}

void measureFunc(kernel_ptr_type func, int size, dim3 blockSize) {
  MeasurementSeries time;

  dim3 grid = size / blockSize.x + 1;

  func<<<grid, blockSize>>>(dA, dB, size);

  int iters = min(30000, max(2000, 100000 * 10000 / size));

  if (usePersistentThreadsGsync) {

    cudaDeviceProp prop;
    int deviceId;
    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    std::string deviceName = prop.name;
    int smCount = prop.multiProcessorCount;
    int maxActiveBlocks = 0;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, sync_kernel_gsync<double>,
        blockSize.x * blockSize.y * blockSize.z, 0));

    const int blockCount = min(size / (blockSize.x * blockSize.y * blockSize.z),
                               smCount * maxActiveBlocks);

    for (int iter = 0; iter < 3; iter++) {

      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();

      void *kernelArgs[] = {&dA, &dB, &size, &iters};
      GPU_ERROR(cudaLaunchCooperativeKernel((void *)sync_kernel_gsync<double>,
                                            blockCount, blockSize, kernelArgs,
                                            0, 0));

      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
  } else if (usePersistentThreadsAtomic) {

    int *flags;

    GPU_ERROR(cudaMalloc(&flags, sizeof(int) * iters));

    cudaDeviceProp prop;
    int deviceId;
    GPU_ERROR(cudaGetDevice(&deviceId));
    GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
    std::string deviceName = prop.name;
    int smCount = prop.multiProcessorCount;
    int maxActiveBlocks = 0;
    GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActiveBlocks, sync_kernel_atomic<double>,
        blockSize.x * blockSize.y * blockSize.z, 0));

    const int blockCount = min(size / (blockSize.x * blockSize.y * blockSize.z),
                               smCount * maxActiveBlocks);

    for (int iter = 0; iter < 3; iter++) {
      init_kernel<<<52, 256>>>(flags, iters, 0);

      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();

      sync_kernel_atomic<double>
          <<<blockCount, blockSize>>>(flags, dA, dB, size, iters);

      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
    GPU_ERROR(cudaFree(flags));

  } else if (useCudaGraph) {

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    cudaStream_t stream;
    GPU_ERROR(cudaStreamCreate(&stream));

    GPU_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < iters; i += 2) {
      func<<<grid, blockSize, 0, stream>>>(dA, dB, size);
      func<<<grid, blockSize, 0, stream>>>(dB, dA, size);
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

    func<<<grid, blockSize, 0>>>(dA, dB, size);
    for (int iter = 0; iter < 3; iter++) {
      GPU_ERROR(cudaDeviceSynchronize());
      double t1 = dtime();
      for (int i = 0; i < iters; i += 2) {
        func<<<grid, blockSize, 0>>>(dA, dB, size);
        func<<<grid, blockSize, 0>>>(dB, dA, size);
      }
      GPU_ERROR(cudaDeviceSynchronize());
      double t2 = dtime();
      time.add((t2 - t1) / iters);
    }
  }
  cout << fixed << setprecision(0) << setw(6) << setw(5)
       << size * 2 * sizeof(double) / time.median() * 1e-9 << "  ";
  ;
  cout.flush();
}

int main(int argc, char **argv) {
  if (argc > 1 && string(argv[1]) == "-graph") {
    useCudaGraph = true;
  }
  if (argc > 1 && string(argv[1]) == "-pta") {
    usePersistentThreadsAtomic = true;
  }
  if (argc > 1 && string(argv[1]) == "-pt-gsync") {
    usePersistentThreadsGsync = true;
  }

  GPU_ERROR(cudaMalloc(&dA, max_buffer_count * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_count * sizeof(double)));

  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dA, max_buffer_count,
                                                     0.23);
  init_kernel<<<max_buffer_count / 1024 + 1, 1024>>>(dB, max_buffer_count,
                                                     1.44);

  GPU_ERROR(cudaDeviceSynchronize());

  for (int d = 4 * 1024; d < 8 * 16 * 1024 * 1024;
       d += std::max((int)1, (int)(d * 0.06))) {

    std::cout << d << "  " << d * sizeof(double) * 2 / 1024 << "kB  ";

    for (int xblock = 32; xblock <= 1024; xblock *= 2) {
      measureFunc(scale<double>, d, dim3(xblock, 1, 1));
    }
    std::cout << "\n";
    std::cout.flush();
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
