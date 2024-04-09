#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-clock.cuh"
#include "../gpu-error.h"
#include "../gpu-metrics/gpu-metrics.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unistd.h>
#ifdef __NVCC__
#include <nvml.h>
#endif
#ifdef __HIP__
#include <rocm_smi/rocm_smi.h>
#endif

using namespace std;

const size_t max_buffer_size = (size_t)1024 * 1024 * 256;
char *dA, *dB;

template <typename T, int N>
using kernel_ptr_type = void (*)(T *A, T *B, int zero, int one);

unsigned int gpu_clock = 0;

template <typename T>
__global__ void initKernel(T *A, const T *__restrict__ B, const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = T(0.1);
  }
}

template <typename T, int N>
__global__ __launch_bounds__(1024) void blockKernel(T *A, T *B, int zero,
                                                    int pitch) {
  int tidx = threadIdx.x % 64;
  int tidy = threadIdx.y % 32;

  T sum = T(0);
  T *A2 = A + tidx + tidy * pitch + pitch;

#pragma unroll 1
  for (int n = 0; n < N; n += 8) {
    A2 += zero;
    sum += A2[0] * A2[4];
    sum += A2[8] * A2[12];
    sum += A2[20] * A2[16];
    sum += A2[24] * A2[28];
  }
  if (sum == T(123123.23))
    B[tidx] = sum;
}

template <typename T, int N>
__global__ void strideKernel(T *A, T *B, int zero, int stride) {
  int tidx = threadIdx.x;

  T sum = T(0);
  T *A2 = A + (tidx % 64) * stride;

#pragma unroll 1
  for (int n = 0; n < N; n += 8) {
    A2 += zero;
    sum += A2[0] * A2[4];
    sum += A2[8] * A2[12];
    sum += A2[20] * A2[16];
    sum += A2[24] * A2[28];

    // T v3 = A2[17] + A2[20] * A2[22];
    // T v4 = A2[24] + A2[26] * A2[30];
    // T v5 = A2[32] + A2[35] * A2[38];
    // T v6 = A2[40] + A2[43] * A2[45];

    // v1 = v1 + v2 * v3;
    // v5 = v4 + v5 * v6;
  }

  if (sum == T(123123.23))
    B[tidx] = sum;
}

#ifdef __HIPCC__
std::vector<const char *> metrics = {
    "ZERO", "ZERO" //, "GL2C_HIT_sum", "GL2C_MISS_sum"
                   //"TA_TOTAL_WAVEFRONTS_sum",
                   //"TA_UTIL", "TA_BUSY_avr"
                   // "TA_ADDR_STALLED_BY_TC_CYCLES_sum",
                   // "TA_ADDR_STALLED_BY_TD_CYCLES_sum",
    //"TA_DATA_STALLED_BY_TC_CYCLES_sum", "TA_FLAT_WAVEFRONTS_sum",
    //"TA_FLAT_READ_WAVEFRONTS_sum", "TA_BUFFER_WAVEFRONTS_sum",
    // "TA_BUFFER_READ_WAVEFRONTS_sum",
    // "TA_BUFFER_WRITE_WAVEFRONTS_sum",
    //"TA_BUFFER_TOTAL_CYCLES_sum", "TA_BUFFER_COALESCED_READ_CYCLES_sum",
    //"TA_BUFFER_COALESCED_WRITE_CYCLES_sum",
    //"TD_TD_BUSY_sum",
    // "TD_TC_STALL_sum",
    //"TD_LOAD_WAVEFRONT_sum", "TD_COALESCABLE_WAVEFRONT_sum",
    // "TD_SPI_STALL_sum",
    // "TCP_GATE_EN1_sum",
    // "TCP_GATE_EN2_sum",
    // "TCP_TD_TCP_STALL_CYCLES_sum",
    // "TCP_TCR_TCP_STALL_CYCLES_sum",
    // "TCP_READ_TAGCONFLICT_STALL_CYCLES_sum",
    // "TCP_VOLATILE_sum",
    // "TCP_TOTAL_ACCESSES_sum",
    //  "TCP_TOTAL_READ_sum",
    // "TCP_TOTAL_WRITE_sum",
    // "TCP_TOTAL_WRITEBACK_INVALIDATES_sum",
    // "TCP_UTCL1_REQUEST_sum", "TCP_UTCL1_TRANSLATION_MISS_sum",
    //"TCP_UTCL1_TRANSLATION_HIT_sum",
    //"TCP_UTCL1_PERMISSION_MISS_sum",
    // "TCP_TOTAL_CACHE_ACCESSES_sum",
    // "TCP_TCP_LATENCY_sum",
    //"TCP_TA_TCP_STATE_READ_sum",
    // "TCP_TCC_READ_REQ_LATENCY_sum",
    // "TCP_TCC_WRITE_REQ_LATENCY_sum",
    // "TCP_TCC_READ_REQ_sum", "TCP_TCC_WRITE_REQ_sum",
    // "TCP_TCC_NC_READ_REQ_sum",
    // "TCP_TCC_UC_READ_REQ_sum",
    // "TCP_TCC_CC_READ_REQ_sum",
    // "TCP_TCC_RW_READ_REQ_sum",
    //"TCP_PENDING_STALL_CYCLES_sum"
};
#else
std::vector<const char *> metrics = {
    //"l1tex__cycles_active.sum",
    "l1tex__data_pipe_lsu_wavefronts.sum",
    //"l1tex__data_pipe_tex_wavefronts.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum",
    //"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    //"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    //"l1tex__f_wavefronts.sum",
    //"l1tex__lsu_writeback_active.sum",
    //"l1tex__tex_writeback_active.sum",
    //"l1tex__lsuin_requests.sum", "l1tex__data_bank_reads.sum",
    //"l1tex__data_bank_writes.sum",
    //"l1tex__m_l1tex2xbar_req_cycles_active.sum",
    //"l1tex__m_xbar2l1tex_read_sectors.sum",
    //"l1tex__texin_sm2tex_req_cycles_active.sum",
    "lts__t_sectors.sum"};
#endif

template <typename T, int N>
std::vector<double> measureFunc(kernel_ptr_type<T, N> kernel, dim3 blockSize,
                                int arg) {

  int deviceId;
  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  int warpSize = 32; // prop.warpSize;  // normalize to 32 for all GPU types

  std::vector<double> results;

  dim3 grid(1, 1, 1);

  kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);

  MeasurementSeries time;

  GPU_ERROR(cudaDeviceSynchronize());

  for (int iter = 0; iter < 11; iter++) {
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());

    kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    double dt = (t2 - t1);
    time.add(dt);
  }

  results.push_back(time.minValue() * gpu_clock * 1e6 /
                    ((size_t)blockSize.x * blockSize.y * N / warpSize));

  results.push_back(((size_t)blockSize.x * blockSize.y * N * sizeof(T)) /
                    (time.minValue() * gpu_clock * 1e6));
  for (auto metricName : metrics) {
    if (metricName == "ZERO") {
      results.push_back(0);
      continue;
    }
    MeasurementSeries metricSeries;
    GPU_ERROR(cudaDeviceSynchronize());
    measureMetricsStart({metricName});
    kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);
    auto res = measureMetricsStop();

    if (res.size() > 0) {
      metricSeries.add(res[0]);
    } else {
      std::cout << "Could not measure " << metricName << "\n";
    }

    results.push_back(metricSeries.median() /
                      (blockSize.x * blockSize.y * N / warpSize));
  }
  return results;
}

template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F &&f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

dim3 getColor(double intensity) {
  return dim3(max(0.0, min(1.0, 0.4 + intensity * 2.0)) * 255,
              max(0.0, min(1.0, 1.4 - abs(intensity - 0.5) * 2.0)) * 255,
              max(0.0, min(1.0, 2.4 - intensity * 2.0)) * 255);
}

int getPrecision(double value) {

  int precision = 1;
  if (value > 50)
    precision = 0;
  if (value < 5.0)
    precision = 2;
  return precision;
}

void format(std::vector<std::vector<double>> values,
            std::vector<string> rowLabels) {

  vector<vector<std::string>> strings(rowLabels.size());
  vector<vector<double>> colors(rowLabels.size());
  std::vector<double> rowMax(rowLabels.size(), -1.0);
  std::vector<double> rowMin(rowLabels.size(), -1.0);

  for (int row = 0; row < (int)strings.size(); row++) {
    strings[row].push_back(rowLabels[row]);
    for (auto &v : values) {
      double value = v[row];
      if (value > rowMax[row] || rowMax[row] == -1)
        rowMax[row] = value;
      if (value < rowMin[row] || rowMin[row] == -1)
        rowMin[row] = value;

      int precision = getPrecision(value);

      stringstream buf;
      buf << fixed << setprecision(precision) << v[row];
      strings[row].push_back(buf.str());
    }
  }

  for (int row = 0; row < (int)strings.size(); row++) {
    colors[row].push_back(0.5);
    for (auto &v : values) {
      double value = v[row];
      colors[row].push_back((value - rowMin[row]) /
                            max(0.01, rowMax[row] - rowMin[row]));
    }
  }
  std::vector<int> columnLengths;

  for (int i = 0; i < (int)strings[0].size(); i++) {
    columnLengths.push_back(0);
    for (auto &d : strings) {
      columnLengths[i] = std::max(columnLengths[i], (int)d[i].size());
    }
  }

  std::cout << "\n";

  for (int row = 0; row < (int)strings.size(); row++) {
    for (int col = 0; col < (int)strings[row].size(); col++) {
      // std::cout << "\033[0;" << 31 + i / 3 << "m";
      //
      // std::cout << setprecision(val < 50 && val - floorf(val) > 0.001 ? 3 :
      // 0)
      //          << fixed << setw(columnLengths[i]) << val << " ";
      //
      dim3 color = getColor(colors[row][col]);
      cout << "\033[38;2;" << color.x << ";" << color.y << ";" << color.z
           << "m";
      std::cout << setw(columnLengths[col]) << strings[row][col] << "\033[0m  ";
    }
    std::cout << "\n";
  }
}

int main(int argc, char **argv) {
  initMeasureMetric();

  GPU_ERROR(cudaMalloc(&dA, max_buffer_size * sizeof(double)));
  GPU_ERROR(cudaMalloc(&dB, max_buffer_size * sizeof(double)));

  initKernel<<<256, 400>>>(dA, dA, max_buffer_size);
  initKernel<<<256, 400>>>(dB, dB, max_buffer_size);
  GPU_ERROR(cudaDeviceSynchronize());

  gpu_clock = getGPUClock();

  vector<vector<double>> values;

  vector<string> rowLabels({"arg", "cycles", "B/cycle"});

  for (auto &metricName : metrics) {
    rowLabels.push_back(metricName);
  }

  const int N = 18 * 64 * 1024;

  cout << "\n";

  for (int i = 0; i <= 64; i++) {
    auto values =
        measureFunc<float, N>(strideKernel<float, N>, dim3(1024, 1, 1), i);
    cout << "float stride   " << setw(3) << i << "  " << setw(3) << i << "  ";
    for (auto v : values) {
      cout << fixed << setprecision(getPrecision(v)) << setw(6) << v << " ";
    }
    cout << "\n";
  }
  cout << "\n";

  for (int i = 0; i <= 64; i++) {
    auto values =
        measureFunc<double, N>(strideKernel<double, N>, dim3(1024, 1, 1), i);
    cout << "double stride   " << setw(3) << i << "  " << setw(3) << i << "  ";
    for (auto v : values) {
      cout << fixed << setprecision(getPrecision(v)) << setw(6) << v << " ";
    }
    cout << "\n";
  }
  cout << "\n";
  for (int w = 1; w <= 64; w *= 2) {
    int pitch = 4096 + 2;
    auto values = measureFunc<float, N>(blockKernel<float, N>,
                                        dim3(w, 1024 / w, 1), pitch);

    cout << "float block   " << setw(3) << w << "  " << setw(3) << pitch
         << "  ";
    for (auto v : values) {
      cout << fixed << setprecision(getPrecision(v)) << setw(6) << v << " ";
    }
    cout << "\n";
  }

  cout << "\n";
  for (int w = 1; w <= 64; w *= 2) {
    int pitch = 4096 + 2;
    auto values = measureFunc<double, N>(blockKernel<double, N>,
                                         dim3(w, 1024 / w, 1), pitch);

    cout << "double block   " << setw(3) << w << "  " << setw(3) << pitch
         << "  ";
    for (auto v : values) {
      cout << fixed << setprecision(getPrecision(v)) << setw(6) << v << " ";
    }
    cout << "\n";
  }

  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
