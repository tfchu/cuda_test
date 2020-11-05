#include <stdio.h>

__global__
void use_local_memory_GPU(float in)
{
    float f;
    f = in;
}

__global__
void use_global_memory_GPU(float *array)
{
    // array is a pointer into global memory on the device 
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

__global__
void use_shared_memory_GPU(float *array)
{   
    // local variables, private to each thread 
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifecycle as the thread block
    __shared__ float sh_arr[128];

    // copy data from "array" in global memory to sh_arr in shared memory
    // here, each thread is responsible for copying a single element
    sh_arr[index] = array[index];

    __syncthreads();    // ensure all writes to shared memory have completed

    // now sh_arr is fully populated, let's find the average of all previous elements
    // e.g. index = 2, then (sh_arr[0] + sh_arr[1]) / (2 + 1.0f)
    for (i = 0; i<index; i++) { sum += sh_arr[i]; }
    average = sum / (index + 1.0f);

    // if array[index] is greater than the average of arrya[0...index-1], replace with average. 
    // since array[] is in global memory, this change will be seen by the host (and potentially other thread blocks if any)
    if (array[index] > average) { array[index] = average; }

    // the following code has no effect. it modifies shared memory, but the resulting modified data is never copied back
    // to global memory and vanishes when the thread block completes
    sh_arr[index] = 3.14;
}

int main(int argc, char **argv)
{
    use_local_memory_GPU<<<1, 128>>>(2.0f);

    /*
    kernel that uses global memory
    */
    float h_arr[128];
    float *d_arr;

    // allocate global memory on the device, place result in d_arr
    cudaMalloc((void **) &d_arr, sizeof(float) * 128);
    // copy data from host memory h_arr to device memory d_arr
    cudaMemcpy((void *)d_arr, (void *)h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
    // launch kernel
    use_global_memory_GPU<<<1, 128>>>(d_arr);
    // copy the modified data back to host, overwriting content of h_arr
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    // do stuff
    for (int i = 0; i < 128; i++) 
    {
        printf("h_arr[%d] = %f\n", i, h_arr[i]);
    }

    /*
    kernel that uses shared memory
    */
    // launch kernel, pass in a pointer to data in global memory
    use_shared_memory_GPU<<<1, 128>>>(d_arr);
    // copy the modified data back to host, overwriting content of h_arr
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    // do stuff
    for (int i = 0; i < 128; i++) 
    {
        printf("h_arr[%d] = %f\n", i, h_arr[i]);
    }

    return 0;
}