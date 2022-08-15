#ifndef __MEASUREMETRICPW_H_
#define __MEASUREMETRICPW_H_

#include "Eval.hpp"
#include "Metric.hpp"
#include <cuda.h>
#include <cupti_profiler_target.h>
#include <cupti_target.h>
#include <functional>
#include <nvperf_host.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
//#include <FileOp.h>

#define NVPW_API_CALL(apiFuncCall)                                             \
  do {                                                                         \
    NVPA_Status _status = apiFuncCall;                                         \
    if (_status != NVPA_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",     \
              __FILE__, __LINE__, #apiFuncCall, _status);                      \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define CUPTI_API_CALL(apiFuncCall)                                            \
  do {                                                                         \
    CUptiResult _status = apiFuncCall;                                         \
    if (_status != CUPTI_SUCCESS) {                                            \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",     \
              __FILE__, __LINE__, #apiFuncCall, _status);                      \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define DRIVER_API_CALL(apiFuncCall)                                           \
  do {                                                                         \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",     \
              __FILE__, __LINE__, #apiFuncCall, _status);                      \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
  do {                                                                         \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",     \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));  \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

static int numRanges = 2;

#endif // __MEASUREMETRICPW_H_

namespace {
CUcontext cuContext;

CUdevice cuDevice;
std::string chipName;
std::vector<std::string> metricNames;

std::vector<uint8_t> counterDataImage;
std::vector<uint8_t> counterDataImagePrefix;
std::vector<uint8_t> configImage;
std::vector<uint8_t> counterDataScratchBuffer;
std::vector<uint8_t> counterAvailabilityImage;
} // namespace

bool CreateCounterDataImage(std::vector<uint8_t> &counterDataImage,
                            std::vector<uint8_t> &counterDataScratchBuffer,
                            std::vector<uint8_t> &counterDataImagePrefix) {

  CUpti_Profiler_CounterDataImageOptions counterDataImageOptions;
  counterDataImageOptions.pCounterDataPrefix = &counterDataImagePrefix[0];
  counterDataImageOptions.counterDataPrefixSize = counterDataImagePrefix.size();
  counterDataImageOptions.maxNumRanges = numRanges;
  counterDataImageOptions.maxNumRangeTreeNodes = numRanges;
  counterDataImageOptions.maxRangeNameLength = 64;

  CUpti_Profiler_CounterDataImage_CalculateSize_Params calculateSizeParams = {
      CUpti_Profiler_CounterDataImage_CalculateSize_Params_STRUCT_SIZE};

  calculateSizeParams.pOptions = &counterDataImageOptions;
  calculateSizeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;

  CUPTI_API_CALL(
      cuptiProfilerCounterDataImageCalculateSize(&calculateSizeParams));

  CUpti_Profiler_CounterDataImage_Initialize_Params initializeParams = {
      CUpti_Profiler_CounterDataImage_Initialize_Params_STRUCT_SIZE};
  initializeParams.sizeofCounterDataImageOptions =
      CUpti_Profiler_CounterDataImageOptions_STRUCT_SIZE;
  initializeParams.pOptions = &counterDataImageOptions;
  initializeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  counterDataImage.resize(calculateSizeParams.counterDataImageSize);
  initializeParams.pCounterDataImage = &counterDataImage[0];
  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitialize(&initializeParams));

  CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params
      scratchBufferSizeParams = {
          CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params_STRUCT_SIZE};
  scratchBufferSizeParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;
  scratchBufferSizeParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  CUPTI_API_CALL(cuptiProfilerCounterDataImageCalculateScratchBufferSize(
      &scratchBufferSizeParams));

  counterDataScratchBuffer.resize(
      scratchBufferSizeParams.counterDataScratchBufferSize);

  CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params
      initScratchBufferParams = {
          CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params_STRUCT_SIZE};
  initScratchBufferParams.counterDataImageSize =
      calculateSizeParams.counterDataImageSize;

  initScratchBufferParams.pCounterDataImage =
      initializeParams.pCounterDataImage;
  initScratchBufferParams.counterDataScratchBufferSize =
      scratchBufferSizeParams.counterDataScratchBufferSize;
  initScratchBufferParams.pCounterDataScratchBuffer =
      &counterDataScratchBuffer[0];

  CUPTI_API_CALL(cuptiProfilerCounterDataImageInitializeScratchBuffer(
      &initScratchBufferParams));

  return true;
}

bool runTestStart(CUdevice cuDevice, std::vector<uint8_t> &configImage,
                  std::vector<uint8_t> &counterDataScratchBuffer,
                  std::vector<uint8_t> &counterDataImage,
                  CUpti_ProfilerReplayMode profilerReplayMode,
                  CUpti_ProfilerRange profilerRange) {

  // DRIVER_API_CALL(cuCtxCreate(&cuContext, 0, cuDevice));

  CUpti_Profiler_BeginSession_Params beginSessionParams = {
      CUpti_Profiler_BeginSession_Params_STRUCT_SIZE};
  CUpti_Profiler_SetConfig_Params setConfigParams = {
      CUpti_Profiler_SetConfig_Params_STRUCT_SIZE};
  CUpti_Profiler_EnableProfiling_Params enableProfilingParams = {
      CUpti_Profiler_EnableProfiling_Params_STRUCT_SIZE};

  beginSessionParams.ctx = NULL;
  beginSessionParams.counterDataImageSize = counterDataImage.size();
  beginSessionParams.pCounterDataImage = &counterDataImage[0];
  beginSessionParams.counterDataScratchBufferSize =
      counterDataScratchBuffer.size();
  beginSessionParams.pCounterDataScratchBuffer = &counterDataScratchBuffer[0];
  beginSessionParams.range = profilerRange;
  beginSessionParams.replayMode = profilerReplayMode;
  beginSessionParams.maxRangesPerPass = numRanges;
  beginSessionParams.maxLaunchesPerPass = numRanges;

  CUPTI_API_CALL(cuptiProfilerBeginSession(&beginSessionParams));

  setConfigParams.pConfig = &configImage[0];
  setConfigParams.configSize = configImage.size();

  setConfigParams.passIndex = 0;
  CUPTI_API_CALL(cuptiProfilerSetConfig(&setConfigParams));
  CUPTI_API_CALL(cuptiProfilerEnableProfiling(&enableProfilingParams));
  return true;
}

bool runTestEnd() {
  CUpti_Profiler_DisableProfiling_Params disableProfilingParams = {
      CUpti_Profiler_DisableProfiling_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerDisableProfiling(&disableProfilingParams));

  CUpti_Profiler_UnsetConfig_Params unsetConfigParams = {
      CUpti_Profiler_UnsetConfig_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerUnsetConfig(&unsetConfigParams));
  CUpti_Profiler_EndSession_Params endSessionParams = {
      CUpti_Profiler_EndSession_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerEndSession(&endSessionParams));

  // DRIVER_API_CALL(cuCtxDestroy(cuContext));

  return true;
}

bool static initialized = false;

double measureMetricStart(std::vector<std::string> newMetricNames) {

  if (!initialized) {
    initialized = true;
    cudaFree(0);
  }
  int deviceNum = 0;
  int computeCapabilityMajor = 0, computeCapabilityMinor = 0;
  DRIVER_API_CALL(cuDeviceGet(&cuDevice, deviceNum));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      cuDevice));
  DRIVER_API_CALL(cuDeviceGetAttribute(
      &computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      cuDevice));
  if (computeCapabilityMajor < 7) {
    printf("Sample unsupported on Device with compute capability < 7.0\n");
    return -2.0;
  }

  metricNames = newMetricNames;
  counterDataImagePrefix = std::vector<uint8_t>();
  configImage = std::vector<uint8_t>();
  counterDataScratchBuffer = std::vector<uint8_t>();
  counterDataImage = std::vector<uint8_t>();

  CUpti_Profiler_Initialize_Params profilerInitializeParams = {
      CUpti_Profiler_Initialize_Params_STRUCT_SIZE};
  CUPTI_API_CALL(cuptiProfilerInitialize(&profilerInitializeParams));
  /* Get chip name for the cuda  device */
  CUpti_Device_GetChipName_Params getChipNameParams = {
      CUpti_Device_GetChipName_Params_STRUCT_SIZE};
  getChipNameParams.deviceIndex = deviceNum;
  CUPTI_API_CALL(cuptiDeviceGetChipName(&getChipNameParams));
  chipName = getChipNameParams.pChipName;

  CUpti_Profiler_GetCounterAvailability_Params getCounterAvailabilityParams = {
      CUpti_Profiler_GetCounterAvailability_Params_STRUCT_SIZE};
  getCounterAvailabilityParams.ctx = cuContext;
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));

  counterAvailabilityImage.clear();
  counterAvailabilityImage.resize(
      getCounterAvailabilityParams.counterAvailabilityImageSize);
  getCounterAvailabilityParams.pCounterAvailabilityImage =
      counterAvailabilityImage.data();
  CUPTI_API_CALL(
      cuptiProfilerGetCounterAvailability(&getCounterAvailabilityParams));
  /* Generate configuration for metrics, this can also be done offline*/
  NVPW_InitializeHost_Params initializeHostParams = {
      NVPW_InitializeHost_Params_STRUCT_SIZE};
  NVPW_API_CALL(NVPW_InitializeHost(&initializeHostParams));
  if (metricNames.size()) {
    if (!NV::Metric::Config::GetConfigImage(chipName, metricNames, configImage,
                                            counterAvailabilityImage.data())) {
      std::cout << "Failed to create configImage" << std::endl;
      return -1.0;
    }
    if (!NV::Metric::Config::GetCounterDataPrefixImage(
            chipName, metricNames, counterDataImagePrefix)) {
      std::cout << "Failed to create counterDataImagePrefix" << std::endl;
      return -1.0;
    }
  } else {
    std::cout << "No metrics provided to profile" << std::endl;
    return -1.0;
  }

  if (!CreateCounterDataImage(counterDataImage, counterDataScratchBuffer,
                              counterDataImagePrefix)) {
    std::cout << "Failed to create counterDataImage" << std::endl;
  }

  CUpti_ProfilerReplayMode profilerReplayMode = CUPTI_KernelReplay;
  CUpti_ProfilerRange profilerRange = CUPTI_AutoRange;

  runTestStart(cuDevice, configImage, counterDataScratchBuffer,
               counterDataImage, profilerReplayMode, profilerRange);
  return 0.0;
}

// double measureMetric(std::function<double()> runPass,
//                      std::vector<std::string> metricNames) {
//   measureMetricStart(metricNames);
//   runPass();
//   return measureMetricStop();
//}

extern "C" void measureBandwidthStart() {
  measureMetricStart({"dram__bytes_read.sum", "dram__bytes_write.sum",
                      "lts__t_sectors_op_read.sum",
                      "lts__t_sectors_op_write.sum"});
}

extern "C" double measureMetricStopPrint() {

  runTestEnd();

  // CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
  //     CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  // CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

  NV::Metric::Eval::PrintMetricValues(chipName, counterDataImage, metricNames);

  return 0.0;
}

std::vector<double> measureMetricStop() {

  runTestEnd();

  // CUpti_Profiler_DeInitialize_Params profilerDeInitializeParams = {
  //     CUpti_Profiler_DeInitialize_Params_STRUCT_SIZE};
  // CUPTI_API_CALL(cuptiProfilerDeInitialize(&profilerDeInitializeParams));

  auto results = NV::Metric::Eval::GetMetricValues(chipName, counterDataImage,
                                                   metricNames);
  return results;
}
void initMeasureMetric(){};
