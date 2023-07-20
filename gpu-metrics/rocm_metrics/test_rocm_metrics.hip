#include <iostream>
#include "rocm_metrics.hpp"


__global__ void updateKernel(double* A, size_t N) {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    for(size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = 0.2 * A[i];;
    }
}


int main(int argc, char** argv) {
    initMeasureMetric();

    double *dA;
    int bufferCount = 1024 * 1024 * 1024;
    hipMalloc(&dA, bufferCount * sizeof(double));

    for(int i = 0; i < 10; i++) {
        measureBandwidthStart();

        updateKernel<<<100, 1024>>>(dA, bufferCount);
        auto vals = measureMetricStop();

        for(auto v: vals) {
            std::cout << v * 1024 / bufferCount << "\n";
        }
    }

    return 0;
}
