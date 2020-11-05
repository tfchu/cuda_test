#include <stdio.h>
#include <cuda_runtime.h>

int log2(int i)
{
    int r = 0;
    while (i >>= 1) r++;
    return r;
}

int bit_reverse(int w, int bits)
{
    int r = 0;
    for (int i = 0; i < bits; i++)
    {
        int bit = (w & (1 << i)) >> i;
        r |= bit << (bits - i - 1);
    }
    return r;
}

/*
Using device 0:
NVIDIA Tegra X1; global mem: 2076037120B; compute v5.3; clock: 921600 kHz
Running naive histo
bin 0: count 7
bin 1: count 7
bin 2: count 6
bin 3: count 6
bin 4: count 7
bin 5: count 6
bin 6: count 7
bin 7: count 6
bin 8: count 7
bin 9: count 7
bin 10: count 7
bin 11: count 7
bin 12: count 7
bin 13: count 6
bin 14: count 6
bin 15: count 8

incorrect due to race condition in d_bins[myBin]++
this does not happen in serial code as each thread runs separately
e.g. BIN with value 5, and thread 1 and 2 wants to increase it
    - thread 1 reads 5, increase to 6, write 6 back to bin
    - thread 2 reads 5, increase to 6, write 6 back to bin
    - but actual answer is 7
*/
__global__ void naive_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    d_bins[myBin]++;
}

/*
Using device 0:
NVIDIA Tegra X1; global mem: 2076037120B; compute v5.3; clock: 921600 kHz
Running simple histo
bin 0: count 4096
bin 1: count 4096
bin 2: count 4096
bin 3: count 4096
bin 4: count 4096
bin 5: count 4096
bin 6: count 4096
bin 7: count 4096
bin 8: count 4096
bin 9: count 4096
bin 10: count 4096
bin 11: count 4096
bin 12: count 4096
bin 13: count 4096
bin 14: count 4096
bin 15: count 4096
*/
__global__ void simple_histo(int *d_bins, const int *d_in, const int BIN_COUNT)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int myItem = d_in[myId];
    int myBin = myItem % BIN_COUNT;
    atomicAdd(&(d_bins[myBin]), 1);
}


int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);

    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n",
               devProps.name, (int)devProps.totalGlobalMem, 
               (int)devProps.major, (int)devProps.minor, 
               (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 65536;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    const int BIN_COUNT = 16;
    const int BIN_BYTES = BIN_COUNT * sizeof(int);

    // generate the input array on the host
    int h_in[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = bit_reverse(i, log2(ARRAY_SIZE));
    }
    int h_bins[BIN_COUNT];
    for(int i = 0; i < BIN_COUNT; i++) {
        h_bins[i] = 0;
    }

    // declare GPU memory pointers
    int * d_in;
    int * d_bins;

    // allocate GPU memory
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_bins, BIN_BYTES);

    // transfer the arrays to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_bins, h_bins, BIN_BYTES, cudaMemcpyHostToDevice); 

    int whichKernel = 0;
    if (argc == 2) {
        whichKernel = atoi(argv[1]);
    }
        
    // launch the kernel
    switch(whichKernel) {
    case 0:
        printf("Running naive histo\n");
        naive_histo<<<ARRAY_SIZE / 64, 64>>>(d_bins, d_in, BIN_COUNT);
        break;
    case 1:
        printf("Running simple histo\n");
        simple_histo<<<ARRAY_SIZE / 64, 64>>>(d_bins, d_in, BIN_COUNT);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }

    // copy back the sum from GPU
    cudaMemcpy(h_bins, d_bins, BIN_BYTES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < BIN_COUNT; i++) {
        printf("bin %d: count %d\n", i, h_bins[i]);
    }

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_bins);
        
    return 0;
}