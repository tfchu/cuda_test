/*
sum of 2^20 elements in 2 stages
stage 1: 
    - launch 1024 blocks, each uses 1024 threads to reduce 1024 elements
    - each block produce 1 single item, so it ends up with 1024 items
stage 2: 
    - launch 1 block to reduce the final 1024 elements into 1 single elements
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// global memory access (R|W)
// loop 0 ~ 9: 
// 1024 | 512, 512 | 256, 256 | 128, 128 | 64, 64 | 32, 32 | 16, 16 | 8, 8 | 4, 4 | 2, 2 | 1
// total around 3069 R|W
__global__ void global_reduce_kernel(float * d_out, float * d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;   // thread id among all blocks
    int tid  = threadIdx.x;                             // thread id in a block

    // do reduction in global mem
    // for all blocks, do
    // s=512: d_in[0] = d_in[0] + d_in[512], d_in[1] = d_in[1] + d_in[513], ... 
    // s=256: d_in[0] = d_in[0] + d_in[256], d_in[1] = d_in[1] + d_in[257], ...
    // ...
    // s=1: d_in[0] = d_in[0] + d_in[1]
    // d_in[0], d_in[1024], ... are sum of all elements in a thread block
    // now we have 1024 elements
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)   // s=512, 256, 128, 64, 32, 16, 8, 4, 2, 1
    {
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];               // e.g. d_in[0] = d_in[0] + d_in[512]
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    // d_out[0] = d_in[0]
    // d_out[1] = d_in[1024]
    // ...
    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }
}

// global memory access (R), write to shared memory
// pre-loop: 1024
__global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

// global memory kernel is around 3 times more global memory access
// but shared memory kernel does not run 3 times faster, due to
// - we are not saturating memory system
// - other techniques required to max out performance of reduce (micro-optimization)
//   - e.g. process multiple items per thread intead of just 1
//   - e.g. perform 1st step of the reduction right when you read from global memory into shared memory
//   - e.g. take advantage of "warp are synchronous" when doing last step of reduction
void reduce(float * d_out, float * d_intermediate, float * d_in, 
            int size, bool usesSharedMemory)
{
    // assumes that size is not greater than maxThreadsPerBlock^2
    // and that size is a multiple of maxThreadsPerBlock
    // stage 1: 2^20 (1024*1024) -> 1024 elements
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;           // 1024
    int blocks = size / maxThreadsPerBlock;     // 2^20 / 1024 = 1024
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_intermediate, d_in);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_intermediate, d_in);
    }
    // now we're down to one block left, so reduce it
    // stage 2: 1024 -> 1 element
    threads = blocks;   // 1024, launch one thread for each block in prev step
    blocks = 1;         // 1
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>
            (d_out, d_intermediate);
    }
    else
    {
        global_reduce_kernel<<<blocks, threads>>>
            (d_out, d_intermediate);
    }
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

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    // generate the input array on the host
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for(int i = 0; i < ARRAY_SIZE; i++) {
        // generate random float in [-1.0f, 1.0f]
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);
        sum += h_in[i];
    }

    // declare GPU memory pointers
    float * d_in, * d_intermediate, * d_out;

    // allocate GPU memory
    // (void **): a pointer to a pointer to memory with an unspecified type, similar to (int *)
    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES); // overallocated
    cudaMalloc((void **) &d_out, sizeof(float));

    // transfer the input array to the GPU
    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice); 

    int whichKernel = 0;    // 0: global, 1: shared memory
    if (argc == 2) {
        whichKernel = atoi(argv[1]);
    }
        
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // launch the kernel
    switch(whichKernel) {
    case 0:
        printf("Running global reduce\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
        }
        cudaEventRecord(stop, 0);
        break;
    case 1:
        printf("Running reduce with shared mem\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
        }
        cudaEventRecord(stop, 0);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);    
    elapsedTime /= 100.0f;      // 100 trials

    // copy back the sum from GPU
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    printf("average time elapsed: %f\n", elapsedTime);
    printf("final value: %f\n", h_out);

    // free GPU memory allocation
    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
        
    return 0;
}