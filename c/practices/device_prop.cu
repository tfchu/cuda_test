#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

void getCudaDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    for (int i = 0; i < nDevices; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device Number: %d\n", i);
        printf("  Device name: %s\n", prop.name);
        printf("  Memory Clock Rate (KHz): %d\n",
               prop.memoryClockRate);
        printf("  Memory Bus Width (bits): %d\n",
               prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
               2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Multi-processor count: %d\n", prop.multiProcessorCount);
        printf("Maximum size of each dimension of a grid: %d\n", prop.maxGridSize);
        printf("Maximum size of each dimension of a block: %d\n", prop.maxThreadsDim);
        printf("Maximum number of threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Maximum number of resident blocks per multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor);
        printf("Maximum resident threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor); 
      }
}

int main(void)
{
    // GPU info
    getCudaDeviceInfo();
}