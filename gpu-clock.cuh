#include "dtime.hpp"
#include "gpu-error.h"
#include <unistd.h>
#include <iostream>

#ifdef __NVCC__
#include <nvml.h>
#endif
#ifdef __HIP__
#include <rocm_smi/rocm_smi.h>
#endif


__global__ void powerKernel(double* A, int iters) {
    int tidx = threadIdx.x + blockIdx.x*blockDim.x;

    double start = A[0];
    #pragma unroll 1
    for(int i = 0; i < iters; i++) {
        start -= (tidx*0.1)*start;
    }
    A[0] = start;
}



unsigned int getGPUClock() {

    double* dA = NULL;
#ifdef __NVCC__
    GPU_ERROR(cudaMalloc(&dA, sizeof(double)));
#endif
#ifdef __HIP__
    GPU_ERROR(hipMalloc(&dA, sizeof(double)));
#endif

    unsigned int gpu_clock;
  int iters = 10;
  double dt = 0;
  std::cout << "clock: ";
  while (dt < 0.3) {
#ifdef __NVCC__
    GPU_ERROR(cudaDeviceSynchronize());
#endif
#ifdef __HIP__
    GPU_ERROR(hipDeviceSynchronize());
#endif
    double t1 = dtime();

    powerKernel<<<100, 1024>>>(dA, iters);
    usleep(10000);

#ifdef __NVCC__
    nvmlInit();
    nvmlDevice_t device;
    nvmlDeviceGetHandleByIndex(0, &device);
    nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &gpu_clock);
    GPU_ERROR(cudaDeviceSynchronize());
#endif
#ifdef __HIP__
  int deviceId;
    GPU_ERROR(hipGetDevice(&deviceId));
    rsmi_status_t ret;
    uint32_t num_devices;
    uint16_t dev_id;
    rsmi_frequencies_t clockStruct;
    ret = rsmi_init(0);
    ret = rsmi_num_monitor_devices(&num_devices);
    ret = rsmi_dev_gpu_clk_freq_get(deviceId, RSMI_CLK_TYPE_SYS, &clockStruct);
    gpu_clock = clockStruct.frequency[clockStruct.current] / 1e6;
    GPU_ERROR(hipDeviceSynchronize());
#endif
    double t2 = dtime();
    std::cout << gpu_clock << " ";
    std::cout.flush();
    dt = t2 - t1;
    iters *= 2;
  }
  std::cout << "\n";
#ifdef __NVCC__
    GPU_ERROR(cudaFree(dA));
#endif
#ifdef __HIP__
    GPU_ERROR(hipFree(dA));
#endif
  return gpu_clock;
}
