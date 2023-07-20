#ifndef GPU_MEASURE_METRICS_H_
#define GPU_MEASURE_METRICS_H_


#ifdef __NVCC__
#include "cuda_metrics/measureMetricPW.hpp"
#elif defined __HIP__
#include "rocm_metrics/rocm_metrics.hpp"
#endif
#endif // GPU_MEASURE_METRICS_H_
