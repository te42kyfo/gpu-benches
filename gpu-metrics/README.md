# Performance Counter Measurement Library for AMD and NVIDIA GPUs

This folder contains a header that provides pairs of functions:

```
void measureMetricsStart(std::vector<const char *> metricNames);
std::vector<double> measureMetricsStop();
```
The second function will return the measured metrics specicied in the start function of a GPU kernel launched in between the two. Launch only a single GPU kernel, otherwise it will probably crash.
For metricNames, any metric supported by your GPU can be used. Multiple metrics can be measured at the same time. The NVIDIA backend does multi pass if all metrics cannot be profiled in a single pass, the rocprofiler backend crashes but suggestes a different metric combination.

There are two more pairs of start/stop function

```
void measureDRAMBytesStart();
std::vector<double> measureDRAMBytesStop()

void measureL2BytesStart();
void measureL2BytesStop();
```
which contain the metric names and evaluation for very selected GPU models. On the AMD side, should work and tested for gfx90a and gfx1030, on the NVIDIA side, sm_80 aka A100. Dont forget to call

```
void initMeasureMetric();
```
before doing anything.

Example usage from gpu-l2-cache/main.cu: (where it is currently commented out because it doesn't work on all models.

```
measureDRAMBytesStart();
callKernel<N, blockSize>(blockCount, blockRun);
auto metrics = measureDRAMBytesStop();
dram_read.add(metrics[0]);
dram_write.add(metrics[1]);

measureL2BytesStart();
callKernel<N, blockSize>(blockCount, blockRun);
metrics = measureL2BytesStop();
L2_read.add(metrics[0]);
L2_write.add(metrics[1]);
```

The APIs (perf works and rocprofiler) are unstable and fragile, if something is slightly off. Issues and comments welcome. 





