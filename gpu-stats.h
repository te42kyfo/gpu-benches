#ifndef GPU_STATS_H_
#define GPU_STATS_H_

#ifdef __NVCC__
#include <nvml.h>
#elif defined __HIP__
#include <rocm_smi/rocm_smi.h>
#endif

struct GPU_stats {
  double clock;
  double power;
  double temperature;
};

GPU_stats getGPUStats(int deviceId) {
#ifdef __NVCC__
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    nvmlInit();
  }
  nvmlDevice_t device;
  nvmlDeviceGetHandleByIndex(deviceId, &device);
  unsigned int power = 0;
  unsigned int clock = 0;
  unsigned int temperature = 0;

  nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clock);
  nvmlDeviceGetPowerUsage(device, &power);
  nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temperature);

  return {clock, power, temperature};
#elif defined __HIP__

  static bool initialized = false;
  rsmi_status_t ret;
  if (!initialized) {
    initialized = true;
    ret = rsmi_init(0);
    unsigned int num_devices;
    ret = rsmi_num_monitor_devices(&num_devices);
  }

  uint64_t power = 0;
  rsmi_frequencies_t clockStruct;
  int currentClock = 0;
  int64_t temperature = 0;
  ret = rsmi_dev_temp_metric_get(deviceId, RSMI_TEMP_TYPE_EDGE,
                                 RSMI_TEMP_CURRENT, &temperature);
  ret = rsmi_dev_power_ave_get(deviceId, 0, &power);
  ret = rsmi_dev_gpu_clk_freq_get(deviceId, RSMI_CLK_TYPE_SYS, &clockStruct);

  power /= 1000;
  temperature /= 1000;
  currentClock = clockStruct.frequency[clockStruct.current] / 1e6;

  return {(double)currentClock, (double)power, (double)temperature};
#endif
}

#endif // GPU-STATS_H_
