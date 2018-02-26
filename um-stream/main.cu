#include <iomanip>
#include <iostream>
#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
using namespace std;

__global__ void scale(double* A, double* B, size_t N)
{
    size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
	A[i] = 2.1 * B[i];
    }
}

__global__ void triad(double* A, double* B, double* C, size_t N)
{
    size_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
	A[i] = 2.1 * B[i] + C[i];
    }
}

int main(int argc, char** argv)
{
  double *A, *B, *C;

    for (size_t N = 1024; N < (size_t)1024 * 1024 * 1024 * 4; N *= 2) {
	GPU_ERROR(cudaMallocManaged(&A, N * sizeof(double)));
	GPU_ERROR(cudaMallocManaged(&B, N * sizeof(double)));
    GPU_ERROR(cudaMallocManaged(&C, N * sizeof(double)));

    triad<<<640, 256>>>(A, B, C, N);
    //	scale<<<640, 256>>>(A, B, N);
	GPU_ERROR(cudaDeviceSynchronize());

	MeasurementSeries time;
	for (int i = 0; i < 5; i++) {
	    double t1 = dtime();
	    triad<<<640, 256>>>(A, B, C, N);
	    GPU_ERROR(cudaDeviceSynchronize());
	    double t2 = dtime();
	    time.add(t2 - t1);
	}

	double bw = N * sizeof(double) * 3 / time.value() * 1.0e-9;
	cout << fixed << setprecision(1) << N * sizeof(double) * 1e-9 << "  "
	     << time.value() * 1000 << "ms  " << setprecision(1)
	     << time.spread() * 100 << "%  " << bw << "\n";

	GPU_ERROR(cudaFree(A));
	GPU_ERROR(cudaFree(B));
    GPU_ERROR(cudaFree(C));
    }
}
