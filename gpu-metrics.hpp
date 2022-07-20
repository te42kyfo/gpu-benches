#ifndef GPU_MEASURE_METRICS_H_
#define GPU_MEASURE_METRICS_H_


#ifdef __NVCC__
#include "measure_metric/measureMetricPW.hpp"
#elif defined __HIP__
#include "rocm-metrics/rocm-metrics.hpp"
#endif
#endif // GPU_MEASURE_METRICS_H_
