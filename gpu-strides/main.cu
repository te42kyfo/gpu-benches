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

/*float/double
read/write
alignment

uniform
unit stride
stride
block stride

single block
*/

template <typename T, int N>
__global__ void uniformKernel(T *A, T *B, int zero, int alignment) {
  int tidx = threadIdx.x;

  T sum = T(0);
  T *A2 = A + tidx * zero + alignment;

  for (int n = 0; n < N; n += 10) {
    if (sum == 1232)
      A2 += zero * n;
    sum += A2[1 * 32] * A2[2 * 32];
    sum += A2[4 * 32] * A2[3 * 32];
    sum += A2[5 * 32] * A2[6 * 32];
    sum += A2[8 * 32] * A2[7 * 32];
    sum += A2[9 * 32] * A2[10 * 32];
  }
  if (sum == T(123123.23))
    B[tidx] = sum;
}

template <typename T, int N>
__global__ void uniformStrideKernel(T *A, T *B, int zero, int alignment) {
  int tidx = threadIdx.x;

  T sum = T(0);
  T *A2 = A + tidx + alignment;

  for (int n = 0; n < N; n += 10) {
    if (sum == 1232)
      A2 += zero * n;
    sum += A2[1] * A2[2];
    sum += A2[4] * A2[3];
    sum += A2[5] * A2[6];
    sum += A2[8] * A2[7];
    sum += A2[9] * A2[10];
  }
  if (sum == T(123123.23))
    B[tidx] = sum;
}

template <typename T, int N>
__global__ void blockKernel(T *A, T *B, int zero, int pitch) {
  int tidx = threadIdx.x;
  int tidy = threadIdx.y % 64;

  T sum = T(0);
  T *A2 = A + tidx + tidy * pitch;
#pragma unroll 1
  for (int n = 0; n < N; n += 8) {
    A2 += zero;
    sum += A2[8] * A2[16];
    sum += A2[48] * A2[32];
    sum += A2[64] * A2[96];
    sum += A2[128] * A2[0];
  }

  if (sum == T(123123.23))
    B[tidx] = sum;
}

template <typename T, int N>
__global__ void stencilKernel(T *A, T *B, int zero, int pitch) {
  int tidx = threadIdx.x;
  int tidy = threadIdx.y % 64;

  T sum = T(0);
  T *A2 = A + tidx + tidy * pitch + pitch;

  // #pragma unroll 1
  for (int n = 0; n < N; n += 2) {
    A2 += zero;
    sum += A2[0] * A2[4]; //  + A+ A2[-1 - pitch] + A2[0 - pitch] +
                          // A2[1 - pitch] + A2[-1 + pitch] + A2[0 +
                          // pitch] + A2[1 + pitch];
  }

  if (sum == T(123123.23))
    B[tidx] = sum;
}

#ifdef __HIPCC__
std::vector<const char *> metrics = {
    //"TA_TA_BUSY_sum", "TA_TOTAL_WAVEFRONTS_sum",
    "TA_UTIL", "TA_BUSY_avr"
    // "TA_ADDR_STALLED_BY_TC_CYCLES_sum",
    // "TA_ADDR_STALLED_BY_TD_CYCLES_sum",
    // "TA_DATA_STALLED_BY_TC_CYCLES_sum",
    // "TA_FLAT_WAVEFRONTS_sum",
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
    //"TCP_READ_TAGCONFLICT_STALL_CYCLES_sum",
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
    // "TCP_TA_TCP_STATE_READ_sum",
    // "TCP_TCC_READ_REQ_LATENCY_sum",
    // "TCP_TCC_WRITE_REQ_LATENCY_sum",
    //"TCP_TCC_READ_REQ_sum"
    //"TCP_TCC_WRITE_REQ_sum"
    // "TCP_TCC_NC_READ_REQ_sum",
    // "TCP_TCC_UC_READ_REQ_sum",
    // "TCP_TCC_CC_READ_REQ_sum",
    // "TCP_TCC_RW_READ_REQ_sum",
    //"TCP_PENDING_STALL_CYCLES_sum"
};
#else
std::vector<const char *> metrics = {
    "l1tex__cycles_active.sum", "l1tex__data_pipe_lsu_wavefronts.sum",
    //"l1tex__data_pipe_tex_wavefronts.sum",
    "l1tex__t_output_wavefronts_pipe_lsu_mem_global_op_ld.sum",
    //"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum",
    //"l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    //"l1tex__f_wavefronts.sum",
    //"l1tex__lsu_writeback_active.sum",
    //"l1tex__tex_writeback_active.sum",
    //"l1tex__lsuin_requests.sum", "l1tex__data_bank_reads.sum",
    //"l1tex__data_bank_writes.sum",
    "l1tex__m_l1tex2xbar_req_cycles_active.sum",
    "l1tex__m_xbar2l1tex_read_sectors.sum",
    "l1tex__texin_sm2tex_req_cycles_active.sum", "lts__t_sectors.sum"};
#endif

template <typename T, int N>
std::vector<double> measureFunc(kernel_ptr_type<T, N> kernel, dim3 blockSize,
                                int arg) {

  int deviceId;
  cudaDeviceProp prop;
  GPU_ERROR(cudaGetDevice(&deviceId));
  GPU_ERROR(cudaGetDeviceProperties(&prop, deviceId));
  int warpSize = prop.warpSize;

  std::vector<double> results;

  dim3 grid(1, 1, 1);

  kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);

  MeasurementSeries time;

  GPU_ERROR(cudaDeviceSynchronize());

  for (int iter = 0; iter < 11; iter++) {
    double t1 = dtime();
    GPU_ERROR(cudaDeviceSynchronize());

    kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);
    kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);
    kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);
    kernel<<<grid, blockSize>>>((T *)dA, (T *)dB, 0, arg);
    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    double dt = (t2 - t1) / 4;
    time.add(dt);
  }

  results.push_back(time.minValue() * gpu_clock * 1e6 /
                    (blockSize.x * blockSize.y * N / warpSize));

  results.push_back((blockSize.x * blockSize.y * N * sizeof(T)) /
                    (time.minValue() * gpu_clock * 1e6));
  for (auto metricName : metrics) {
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

void format(std::vector<std::vector<double>> values,
            std::vector<string> rowLabels) {

  vector<vector<std::string>> strings(rowLabels.size());
  vector<vector<double>> colors(rowLabels.size());
  std::vector<double> rowMax(rowLabels.size(), -1.0);
  std::vector<double> rowMin(rowLabels.size(), -1.0);

  for (int row = 0; row < strings.size(); row++) {
    strings[row].push_back(rowLabels[row]);
    for (auto &v : values) {
      double value = v[row];
      if (value > rowMax[row] || rowMax[row] == -1)
        rowMax[row] = value;
      if (value < rowMin[row] || rowMin[row] == -1)
        rowMin[row] = value;

      int precision = 1;
      if (value > 50)
        precision = 0;
      if (value < 5.0)
        precision = 2;
      stringstream buf;
      buf << fixed << setprecision(precision) << v[row];
      strings[row].push_back(buf.str());
    }
  }

  for (int row = 0; row < strings.size(); row++) {
    colors[row].push_back(0.5);
    for (auto &v : values) {
      double value = v[row];
      colors[row].push_back((value - rowMin[row]) /
                            max(0.01, rowMax[row] - rowMin[row]));
    }
  }
  std::vector<int> columnLengths;

  for (int i = 0; i < strings[0].size(); i++) {
    columnLengths.push_back(0);
    for (auto &d : strings) {
      columnLengths[i] = std::max(columnLengths[i], (int)d[i].size());
    }
  }

  std::cout << "\n";

  for (int row = 0; row < strings.size(); row++) {
    for (int col = 0; col < strings[row].size(); col++) {
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

  std::cout << "uniform double\n";
  values.clear();
  for (int i = 0; i < 10; i++) {
    values.push_back(measureFunc<double, 10000>(uniformKernel<double, 10000>,
                                                dim3(1024, 1, 1), i));
    values.back().insert(begin(values.back()), i);
  }
  values.clear();
  std::cout << "uniform float\n";
  format(values, rowLabels);
  for (int i = 0; i < 10; i++) {
    values.push_back(measureFunc<float, 10000>(uniformKernel<float, 10000>,
                                               dim3(1024, 1, 1), i));
    values.back().insert(begin(values.back()), i);
  }
  format(values, rowLabels);

  /*
      values.clear();
      for (int i = 0; i < 4; i++) {
        values.push_back(measureFunc<double, 10000>(
            uniformStrideKernel<double, 10000>, dim3(1024, 1, 1), i));
        values.back().insert(begin(values.back()), i);
      }
      for (int i = 0; i < 4; i++) {
        values.push_back(measureFunc<float, 10000>(
            uniformStrideKernel<float, 10000>, dim3(1024, 1, 1), i));
        values.back().insert(begin(values.back()), i);
      }
      format(values, rowLabels);

      values.clear();
      for (int i = 4; i <= 16; i *= 2) {
        for (int pitch = 0; pitch <= 1; pitch++) {
          values.push_back(measureFunc<double, 10000>(
              blockKernel<double, 10000>, dim3(i, 1024 / i, 1), 1000 + pitch));
          values.back().insert(begin(values.back()), i);
        }
      }
      format(values, rowLabels);
    */
  /*
  for (int i = 1; i <= 64; i *= 2) {
    values.clear();
    for (int pitch = 0; pitch <= 4; pitch++) {
      values.push_back(measureFunc<float, 10000>(
          blockKernel<float, 10000>, dim3(i, 1024 / i, 1), 1024 * 16 + pitch));
      values.back().insert(begin(values.back()), pitch);
    }
    std::cout << i << "\n";
    format(values, rowLabels);
  }

  for (int i = 1; i <= 64; i *= 2) {
    values.clear();
    for (int pitch = 0; pitch <= 4; pitch++) {
      values.push_back(measureFunc<double, 10000>(
          blockKernel<double, 10000>, dim3(i, 1024 / i, 1), 1024 * 16 + pitch));
      values.back().insert(begin(values.back()), pitch);
    }
    std::cout << i << "\n";
    format(values, rowLabels);
  }
*/

  std::cout << "\nfloat block strides:\n";
  for (int i = 1; i <= 64; i *= 2) {
    values.clear();
    for (int pitch = 0; pitch <= 16; pitch++) {
      values.push_back(measureFunc<float, 10000>(
          stencilKernel<float, 10000>, dim3(i, 1024 / i, 1), 4096 + pitch));
      values.back().insert(begin(values.back()), pitch);
    }
    std::cout << i << "\n";
    format(values, rowLabels);
  }

  std::cout << "\ndouble block strides:\n";
  for (int i = 1; i <= 64; i *= 2) {
    values.clear();
    for (int pitch = 0; pitch <= 16; pitch++) {
      values.push_back(measureFunc<double, 10000>(
          stencilKernel<double, 10000>, dim3(i, 1024 / i, 1), 4096 + pitch));
      values.back().insert(begin(values.back()), pitch);
    }
    std::cout << i << "\n";
    format(values, rowLabels);
  }
  /*
    for (int i = 1; i <= 64; i *= 2) {
      values.clear();
      for (int pitch = 0; pitch <= 16; pitch++) {
        values.push_back(measureFunc<double, 10000>(stencilKernel<double,
    10000>, dim3(i, 1024 / i, 1), 1024 * 16 + pitch));
        values.back().insert(begin(values.back()), pitch);
      }
      std::cout << i << "\n";
      format(values, rowLabels);
    }*/
  values.clear();
  std::cout << "\n";
  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
}
