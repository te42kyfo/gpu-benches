#include <iomanip>
#include <iostream>
#include "../MeasurementSeries.hpp"
#include "../dtime.hpp"
#include "../gpu-error.h"
using namespace std;

int main(int argc, char** argv) {
    int deviceCount;
    GPU_ERROR(cudaGetDeviceCount(&deviceCount));


    vector<char*> deviceBuffers(deviceCount, nullptr);
    char *host_buffer;
    const size_t buffer_size_bytes = (size_t)2 * 1024 * 1024 * 1024;


    for( int d  = 0; d < deviceCount; d++) {
        GPU_ERROR(cudaSetDevice(d));
        GPU_ERROR(cudaMalloc(& (deviceBuffers[d]), buffer_size_bytes));
        GPU_ERROR(cudaDeviceSynchronize());
    }
    GPU_ERROR(cudaMallocHost(&host_buffer, buffer_size_bytes));


    const int num_streams = 1;
    cudaStream_t streams[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    memset(host_buffer, 0, buffer_size_bytes);

    for (size_t transfer_size_bytes = 2 << 16;
       transfer_size_bytes <= buffer_size_bytes / num_streams;
       transfer_size_bytes *= 16) {

        for(int d = 0; d < deviceCount; d++) {
            GPU_ERROR(cudaSetDevice(d));
            MeasurementSeries time;
            for (int sample = 0; sample < 5; sample++) {
                memset(host_buffer, 0, buffer_size_bytes);
                double t1 = dtime();
                for (int stream = 0; stream < num_streams; stream++) {
                    GPU_ERROR(cudaMemcpyAsync(
                                  deviceBuffers[d] + (size_t)stream * transfer_size_bytes,
                                  host_buffer + (size_t)stream * transfer_size_bytes,
                                  transfer_size_bytes, cudaMemcpyDefault, streams[stream]));
                }

                GPU_ERROR(cudaDeviceSynchronize());
                double t2 = dtime();
                time.add(t2 - t1);
            }
            double bw = num_streams * transfer_size_bytes / time.value();
            cout << fixed  //
                << "Device: " << d << "   "
                 << setw(10) << setprecision(0) << (transfer_size_bytes >> 10)
                 << "kB  "                                                      //
                 << setprecision(2) << setw(7) << time.value() * 1000 << "ms "  //
                 << setprecision(2) << setw(7) << bw * 1e-9 << "GB/s   "        //
                 << time.spread() * 100 << "%\n";
        }
        if(deviceCount > 1) cout << "\n";
    }

    for(int d = 0; d< deviceCount; d++) {
        GPU_ERROR(cudaFree(deviceBuffers[d]));
}
    //  GPU_ERROR(cudaFree(host_buffer));
    GPU_ERROR(cudaFreeHost(host_buffer));
}
