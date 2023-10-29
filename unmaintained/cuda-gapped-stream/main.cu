#include "../MeasurementSeries.hpp"
#include "../measure_metric/measureMetricPW.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include <iomanip>
#include <iostream>
#include <nvml.h>

using namespace std;

const size_t elementCount = 4 * 1024 * 1024 * 1024ull;
double *dA, *dB;

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = 0.1;
  }
}

template <typename T>
__global__ void scale_kernel(T *A, const T *__restrict__ B, int blocks, int spacing) {
  size_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
  if (tidx >= elementCount)
    return;

  size_t idx = ((tidx * spacing) % elementCount + (tidx*spacing) / elementCount) % elementCount;


  T temp = B[idx];

  if(temp == 12223.0 && threadIdx.x > 10000)
      A[idx] = 1.2; // = B[idx] * 1.2;
}

void measureFunc(int blocks, int spacing) {

  MeasurementSeries time;
  int blockSize = 256;
  int gridSize = (elementCount - 1) / blockSize + 1;

  scale_kernel<<<gridSize, blockSize>>>(dA, dB, blocks, spacing);

  nvmlDevice_t device;
  int deviceId;
  cudaGetDevice(&deviceId);
  nvmlDeviceGetHandleByIndex(deviceId, &device);

  for (int iter = 0; iter < 7; iter++) {
    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());
    scale_kernel<<<gridSize, blockSize>>>(dA, dB, blocks, spacing);
    scale_kernel<<<gridSize, blockSize>>>(dA, dB, blocks, spacing);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add((t2 - t1) / 2);
  }

  measureMetricStart({"dram__bytes_read.sum", "dram__bytes_write.sum"});
  scale_kernel<<<gridSize, blockSize>>>(dA, dB, blocks, spacing);
  GPU_ERROR(cudaDeviceSynchronize());
  auto dram_metrics = measureMetricStop();

  measureMetricStart({"lts__t_sectors_srcunit_tex.sum",
                      "lts__t_sectors_srcunit_ltcfabric.sum",
                      "lts__t_sectors.sum"});
  scale_kernel<<<gridSize, blockSize>>>(dA, dB, blocks, spacing);
  GPU_ERROR(cudaDeviceSynchronize());
  auto l2_metrics = measureMetricStop();

  measureMetricStart({"lts__t_tag_requests.sum",
          "lts__t_tag_requests.avg.pct_of_peak_sustained_elapsed"});

  scale_kernel<<<gridSize, blockSize>>>(dA, dB, blocks, spacing);
  GPU_ERROR(cudaDeviceSynchronize());
  auto tag_requests = measureMetricStop();

  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  std::string deviceName = prop.name;
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, scale_kernel<double>, blockSize, 0));


  cout << fixed << setprecision(0)
       << maxActiveBlocks << " "
       << setw(2) << " " << setw(5)
       << blocks << " " << setw(5)
       << spacing << "       eff:  "
       << elementCount  * sizeof(double) / time.value() * 1e-9 << " GB/s "
       << setprecision(0) << setw(8) << dram_metrics[0] / time.value() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(8) << l2_metrics[0]*32 / time.value() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(8) << l2_metrics[1]*32 / time.value() / 1.0e9 << " GB/s "    //
       << setprecision(0) << setw(8) << l2_metrics[2]*32 / time.value() / 1.0e9 << " GB/s "   //
       << setprecision(0) << setw(8) << tag_requests[0] / time.value() / 1.41e9 << " /cyc "   //
       << setprecision(0) << setw(8) << tag_requests[1] << " % ";   //
                                                                                             //
  cout << "   "  << setprecision(2) << setw(5) << dram_metrics[0] / (elementCount * sizeof(double)) << "  ";
  cout << "   "  << setprecision(2) << setw(5) << l2_metrics[0]*32 / (elementCount * sizeof(double)) << "  ";
  cout << "   "  << setprecision(2) << setw(5) << l2_metrics[1]*32 / (elementCount * sizeof(double)) << "  ";
  cout << "   "  << setprecision(2) << setw(5) << l2_metrics[2]*32 / (elementCount * sizeof(double)) << " ";
  cout << "   "  << setprecision(3) << setw(5) << tag_requests[0] / (elementCount) << " ";


  cout << std::endl;
}

int main(int argc, char **argv) {
    int maxSpacing = 512 * 1024 * 1024;
    size_t bufferSize = elementCount * sizeof(double);
    nvmlInit();
    //GPU_ERROR(cudaMalloc(&dA, bufferSize));
    GPU_ERROR(cudaMalloc(&dB, bufferSize));

    init_kernel<<<256, 400>>>(dB, dB, dB, dB, elementCount );
    //init_kernel<<<256, 400>>>(dA, dA, dA, dA, elementCount * maxSpacing);
    GPU_ERROR(cudaDeviceSynchronize());

    cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 32);

    for(int blocks = 1; blocks <= 1; blocks *=2) {
        for(int spacing = 1; spacing <= maxSpacing; spacing *= 2) {
            measureFunc(blocks, spacing);
        }
    }


    //GPU_ERROR(cudaFree(dA));
    GPU_ERROR(cudaFree(dB));
}
