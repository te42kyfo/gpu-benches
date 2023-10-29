#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
#include "../metrics.cuh"
#include <iomanip>
#include <iostream>

using namespace std;

double *dA, *dB;

using kernel_ptr_type = void (*)(int iters, double *A, const double *B);

template <int N, int UNROLL, bool DOTPRODUCT>
__global__ __launch_bounds__(1024, 1) void kernel(int iters, double *A,
                                                  double *B) {

  int widx = threadIdx.x / 32;
  double sum = 0.0;
#pragma unroll(1)
  for (int w = 0; w < (widx % 5) * 11; w++) {
    sum += w;
  }

  double *dA = A + threadIdx.x;
  double *dB = B + threadIdx.x;

#pragma unroll(1)
  for (int iter = 0; iter < iters; iter++) {
#pragma unroll(UNROLL)
    for (int n = 0; n < N; n++) {
      if (DOTPRODUCT)
        sum += dA[n * 32] * dB[n * 32];
      else
        sum += dA[n * 32];
    }
  }

  if (sum == -12.3) {
    A[threadIdx.x] = sum;
  }
}

double pred(int Iint, int Ild, int Idp, int Nsm, int ClL1) {
  int Nq = ceil((double)Nsm / 4);

  int Tdp = Idp * 4;
  int Tld = Ild * 4;
  int Tint = Iint * 2;
  int TL1lat = 32;
  int TL1thru = ClL1 * Nsm;

  int Ttotal = Tint + max(max(Tld, TL1lat) + Tdp, TL1thru);

  cout << setw(5) << Tdp << " ";
  cout << setw(5) << Tld << " ";
  cout << setw(5) << Tint << " ";


  TL1lat = 32 + (double)TL1thru / Ttotal * 16;
  TL1thru = ClL1 * Nsm * (1.0f + (double)TL1thru / Ttotal) * 0.5f;
  Ttotal = Tint + max(max(Tld, TL1lat) + Tdp, TL1thru);


  
  TL1lat = 32 + (double)TL1thru / Ttotal * 16;
  TL1thru = ClL1 * Nsm * (1.0f + (double)TL1thru / Ttotal) * 0.5f;
  Ttotal = Tint + max(max(Tld, TL1lat) + Tdp, TL1thru);

  string cont = "Tint + ";
  if (TL1thru >= max(Tld, TL1lat) + Tdp) {
    cont += " TL1thru ";
  } else if (TL1thru == max(Tld, TL1lat) + Tdp) {
    cont += " | ";
  } else {
    cont += "( ";
    if (Tld > TL1lat) {
      cont += "Tld";
    } else if (Tld == TL1lat) {
      cont += "TL1lat|Tld";
    } else {
      cont += "TL1lat";
    }
    cont += " + Tdp)";
  }

  cout << cont << "  ";

  return Ttotal;
}

template <int DV, int UNROLL, bool DOTPRODUCT>
void measure(int blockSize, bool concise = false) {

  if (DV % (32 * UNROLL) != 0)
    cout << DV << " % " << 32 * UNROLL << " != 0\n";

  if (DV * 8 * 2 > 128 * 1024)
    cout << DV * 8 * 2 << " > " << 128 * 1024 << "\n";

  if (DV * 8 * 2 < 64 * 1024)
    cout << DV * 8 * 2 << " < " << 64 * 1024 << "\n";

  int blockCount = 1;
  const int N = DV / 32;
  int iters = 100000 / N;

  GPU_ERROR(cudaFuncSetCacheConfig(kernel<N, UNROLL, DOTPRODUCT>,
                                   cudaFuncCachePreferL1));

  MeasurementSeries time;
  for (int i = 0; i < 20; i++) {

    GPU_ERROR(cudaDeviceSynchronize());
    double t1 = dtime();

    kernel<N, UNROLL, DOTPRODUCT><<<blockCount, blockSize>>>(iters, dA, dB);

    GPU_ERROR(cudaDeviceSynchronize());
    double t2 = dtime();
    time.add(t2 - t1);
  }
  GPU_ERROR(cudaGetLastError());

  double spread = (time.median() - time.minValue()) / time.median() * 100;
  double dt = time.minValue();
  double bw = (DOTPRODUCT ? 2 : 1) * DV * iters * sizeof(double) / dt / 1e9;
  double cyc = dt / (DV * iters) * 1.38e9 * 32;

  if (concise) {
    cout << fixed << setprecision(2) << setw(7) << cyc << " ";
  } else {

    cout << fixed << setprecision(2);
    cout << setw(3) << UNROLL << "  "     //
         << setw(8) << dt * 1000 << "   " //
         << setw(8) << spread << "   "    //
         << setw(8) << bw << "   "        //
         << setw(8) << cyc << " -- ";
    // << setw(8)
    //<< (20.0 + max(UNROLL * (DOTPRODUCT ? 8 : 4), 30) + UNROLL * 8) /
    //        UNROLL

    int Iint = 10;
    int Ild = UNROLL * (DOTPRODUCT ? 2 : 1);
    int Idp = UNROLL;
    int ClL1 = Ild * 2;
    int Nsm = max(1, blockSize / 32);
    int Nq = max(1, blockSize / 32 / 4);

    cout << setw(5) << pred(Iint, Ild, Idp, ClL1, Nsm) / UNROLL << " ";

    cout << "\n";
  }
}

int main(int argc, char **argv) {

  size_t maxBufferSize = 1024 * 1024;
  GPU_ERROR(cudaMallocManaged(&dA, sizeof(double) * maxBufferSize));
  GPU_ERROR(cudaMallocManaged(&dB, sizeof(double) * maxBufferSize));
  for (size_t i = 0; i < maxBufferSize; i++) {
    dA[i] = 1.2;
    dB[i] = 1.21;
  }

  bool concise = false;
  const bool dotProduct = false;
  for (int blockSize = 32; blockSize <= 1024; blockSize *= 2) {
    measure<8 * 512, 1, dotProduct>(blockSize, concise);
    measure<8 * 512, 2, dotProduct>(blockSize, concise);
    measure<3 * 2048, 3, dotProduct>(blockSize, concise);
    measure<8 * 512, 4, dotProduct>(blockSize, concise);
    measure<6 * 1024, 6, dotProduct>(blockSize, concise);
    measure<8 * 512, 8, dotProduct>(blockSize, concise);
    measure<9 * 512, 9, dotProduct>(blockSize, concise);
    measure<3 * 2048, 12, dotProduct>(blockSize, concise);
    measure<8 * 512, 16, dotProduct>(blockSize, concise);
    measure<9 * 512, 18, dotProduct>(blockSize, concise);
    measure<6 * 1024, 24, dotProduct>(blockSize, concise);
    measure<27 * 256, 27, dotProduct>(blockSize, concise);
    measure<8 * 512, 32, dotProduct>(blockSize, concise);
    cout << "\n";
  }
  GPU_ERROR(cudaFree(dA));
  GPU_ERROR(cudaFree(dB));
  return 0;
}
