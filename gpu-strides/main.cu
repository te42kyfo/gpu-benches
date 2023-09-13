#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <unistd.h>
#ifdef __NVCC__
#include <nvml.h>
#endif
#ifdef __HIP__
#include <rocm_smi/rocm_smi.h>
#endif

using namespace std;

using dtype = double;
const size_t max_buffer_size = (size_t) 1024 * 1024 * 1024 * 1;
dtype *dA, *dB, *dC, *dD;
using kernel_ptr_type = void (*)(dtype *A, dtype *B, int zero, int one);
unsigned int gpu_clock = 0;

template <typename T>
__global__ void init_kernel(T *A, const T *__restrict__ B, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = T(0.1);
  }
}

template <typename T, int STRIDE, int ITERS>
__global__ void rake_kernel(T *A, T *B, int zero, int one) {
  int tidx = (threadIdx.x + blockIdx.x * blockDim.x) % 64;

  T sum = T(0.0);
  const int N = 1;
#pragma unroll 1

  for (int n = 0; n < N; n++) {
    int ptr = tidx * STRIDE;
    for (int i = 0; i < ITERS; i++) {
      ptr += zero;
      for (int s = 0; s < min(7, STRIDE); s++) {
        sum += A[ptr + s * one] * B[ptr + s * one];
      }
    }
    if (sum == T(123.0)) {
      B[tidx] = T(tidx);
    }
  }
}

template <typename T, int ITERS>
__launch_bounds__(1024, 1) __global__
    void block_kernel(T *A, T *B, int width, int height, int zero, int one) {

  double __shared__ spoiler[6 * 1024];
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y * zero;
  if (tidx >= width || tidy >= height)
    return;

  T sum = T(0.0);

#pragma unroll 1
  for (int i = 0; i < ITERS / 64; i++) {
      int idx = 123 + tidy * width + tidx;
      #pragma unroll 1
      for(int y = 0; y < 64; y++) {
          idx += width;
          //idx += zero;
          sum += A[idx] * B[idx];
      }
  }


  if (sum == T(123.0)) {
    spoiler[threadIdx.x] = sum;
    B[tidx] = spoiler[threadIdx.x / 2];
  }
}
std::vector<const char *> metrics = {
    "width", "block width", "range", "B/cycle/CU", "TA_BUSY_avr",
    "TA_TA_BUSY_sum",
    //"TA_TOTAL_WAVEFRONTS_sum",
    // "TA_ADDR_STALLED_BY_TC_CYCLES_sum",
    // "TA_ADDR_STALLED_BY_TD_CYCLES_sum",
    // "TA_DATA_STALLED_BY_TC_CYCLES_sum",
    // "TA_FLAT_WAVEFRONTS_sum",
    // "TA_FLAT_READ_WAVEFRONTS_sum",
    //"TA_BUFFER_WAVEFRONTS_sum",
    // "TA_BUFFER_READ_WAVEFRONTS_sum",
    // "TA_BUFFER_WRITE_WAVEFRONTS_sum",
    // "TA_BUFFER_TOTAL_CYCLES_sum",
    // "TA_BUFFER_COALESCED_READ_CYCLES_sum",
    // "TA_BUFFER_COALESCED_WRITE_CYCLES_sum",
    "TD_TD_BUSY_sum",
    // "TD_TC_STALL_sum",
    // "TD_LOAD_WAVEFRONT_sum",
    // "TD_COALESCABLE_WAVEFRONT_sum",
    // "TD_SPI_STALL_sum",
    // "TCP_GATE_EN1_sum",
    // "TCP_GATE_EN2_sum",
    // "TCP_TD_TCP_STALL_CYCLES_sum",
    // "TCP_TCR_TCP_STALL_CYCLES_sum",
    //"TCP_READ_TAGCONFLICT_STALL_CYCLES_sum",
    // "TCP_VOLATILE_sum",
    // "TCP_TOTAL_ACCESSES_sum",
    //  "TCP_TOTAL_READ_sum",
    // "TCP_TOTAL_WRITE_sum",
    // "TCP_TOTAL_WRITEBACK_INVALIDATES_sum",
    "TCP_UTCL1_REQUEST_sum", "TCP_UTCL1_TRANSLATION_MISS_sum",
    //"TCP_UTCL1_TRANSLATION_HIT_sum",
    //"TCP_UTCL1_PERMISSION_MISS_sum",
    // "TCP_TOTAL_CACHE_ACCESSES_sum",
    // "TCP_TCP_LATENCY_sum",
    // "TCP_TA_TCP_STATE_READ_sum",
    // "TCP_TCC_READ_REQ_LATENCY_sum",
    // "TCP_TCC_WRITE_REQ_LATENCY_sum",
    // "TCP_TCC_READ_REQ_sum",
    // "TCP_TCC_WRITE_REQ_sum",
    // "TCP_TCC_NC_READ_REQ_sum",
    // "TCP_TCC_UC_READ_REQ_sum",
    // "TCP_TCC_CC_READ_REQ_sum",
    // "TCP_TCC_RW_READ_REQ_sum",
    //"TCP_PENDING_STALL_CYCLES_sum"
};

std::vector<MeasurementSeries> measureFunc(int width, int block_width) {

  std::vector<MeasurementSeries> counters(metrics.size());

  dim3 block_size(block_width, 1*64 / block_width, 1);
  int height = max_buffer_size / width;
  dim3 grid(width / block_size.x + 1, height / block_size.y + 1, 1);

  grid = dim3(1,1,1);
  
  const int ITERS = 16*1024;
  block_kernel<double, ITERS>
      <<<grid, block_size>>>(dA, dB, width, height, 0, 1);

  GPU_ERROR(cudaDeviceSynchronize());
  double t1 = dtime();
  GPU_ERROR(cudaDeviceSynchronize());
  block_kernel<double, ITERS>
      <<<grid, block_size>>>(dA, dB, width, height, 0, 1);
  block_kernel<double, ITERS>
      <<<grid, block_size>>>(dA, dB, width, height, 0, 1);
  GPU_ERROR(cudaDeviceSynchronize());
  double t2 = dtime();

  counters[0].add(width);
  counters[1].add(block_width);
  counters[2].add(block_size.y * 2.0 * width * sizeof(double) * 2 / 1024 /
                  1024);

  counters[3].add((long long)2 * ITERS * block_size.x * block_size.y * sizeof(dtype) /
                  ((t2 - t1) / 2) * 1e-9 / (gpu_clock / 1000.0) / 1);

  for (int i = 4; i < metrics.size(); i++) {
    measureMetricsStart({metrics[i]});
    block_kernel<double, ITERS>
        <<<grid, block_size>>>(dA, dB, width, height, 0, 1);
    auto values = measureMetricStop();
    counters[i].add(values[0] / block_size.x / block_size.y / ITERS / 2 * 64);
  }

  return counters;
}

template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

void columnPrint(std::vector<std::vector<MeasurementSeries>> data,
                 std::vector<const char *> metrics) {
  std::vector<int> columnLengths;

  for (int i = 0; i < metrics.size(); i++) {
    columnLengths.push_back(0);
    for (auto &d : data) {

      columnLengths[i] = std::max(
          columnLengths[i],
          (int)to_string((int)d[i].value()).size() +
              (d[i].value() < 50 && d[i].value() - floorf(d[i].value()) > 0.001
                   ? 4
                   : 0));
    }
  }

  std::cout << "\n";

  for (int n = 0; n < data.size(); n++) {
    for (int i = 0; i < metrics.size(); i++) {
      double val = data[n][i].value();

      // std::cout << "\033[0;" << 31 + i / 3 << "m";
      std::cout << setprecision(val < 50 && val - floorf(val) > 0.001 ? 3 : 0)
                << fixed << setw(columnLengths[i]) << val << " ";
      // std::cout << "\033[0m";
    }
    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  initMeasureMetric();

  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dC, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dD, max_buffer_size * sizeof(double)));

  init_kernel<<<256, 400>>>(dA, dA, max_buffer_size);
  init_kernel<<<256, 400>>>(dB, dB, max_buffer_size);
  init_kernel<<<256, 400>>>(dC, dC, max_buffer_size);
  init_kernel<<<256, 400>>>(dD, dD, max_buffer_size);
  GPU_ERROR(cudaDeviceSynchronize());

  gpu_clock = getGPUClock();

  for (int i = 0; i < metrics.size(); i++) {

    std::cout << "\033[0;" << 31 + i / 3 << "m" << metrics[i] << "  \033[0m";
  }
  std::cout << "\n";

  std::vector<std::vector<MeasurementSeries>> data;

  // constexpr_for<1024 * 2, 1024 * 2 * 16 + 1, 1024 * 2>([&](auto i) {
  for (int width = 256; width <= 1024 * 1024 * 4; width *= 2) {
    data.clear();
    data.push_back(measureFunc(width - 1, 1));
    data.push_back(measureFunc(width - 1, 2));
    data.push_back(measureFunc(width - 1, 4));
    data.push_back(measureFunc(width - 1, 8));
    data.push_back(measureFunc(width - 1, 16));
    data.push_back(measureFunc(width - 1, 32));
    data.push_back(measureFunc(width - 1, 64));
    columnPrint(data, metrics);
    std::cout.flush();
  }

  std::cout << "\n";
  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  GPU_ERROR(cudaFree(dC));
  GPU_ERROR(cudaFree(dD));
}
