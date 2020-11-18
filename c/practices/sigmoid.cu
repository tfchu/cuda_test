/* 
# compile 
$ nvcc -o sigmoid sigmoid.cu

# numpy counterpart
import numpy as np
m = np.array(((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)))
s = 1/(1+np.exp(m))
sd = s*(1-s)
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// kernel of device sigmoid function
__global__ 
void kSigmoid(const int nThreads, float const *input, float *output){
    /*  Computes the value of the sigmoid function f(x) = 1/(1 + e^-x).
     Inputs:
     input: array
     output: array, the results of the computation are to be stored here
    */

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = 1.0 / (1.0 + std::exp(-input[i]));
	  }
}

// cuda version (device-side) of sigmoid function
void dSigmoid(float const *input, float *output, const int height, const int width){

	kSigmoid <<< height, width >>> (height * width, input, output);
	cudaDeviceSynchronize();
}

// kernel of derivative of sigmoid function
__global__ 
void kSigmoid_d(const int nThreads, float const *input, float *output) {
	/*  Computes the value of the sigmoid function derivative f'(x) = f(x)(1 - f(x)),
	    where f(x) is sigmoid function.
	    Inputs:
	    input: array
	    output: array, the results of the computation are to be stored here:
	    		x(1 - x) for every element of the input matrix m1.
	*/

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = input[i] * (1 - input[i]);
	  }
}

// derivative of sigmoid function (d: device, d: derivative)
float* dSigmoid_d(float const *input, float *output, const int rows, const int columns){
	kSigmoid_d <<< rows, columns >>> (rows*columns, input, output);
	cudaDeviceSynchronize();
	return output;
}

int main(void)
{
    // host initialization
    const int M1_SIZE = 12;      // 4x3 matrix
    const int M1_BYTES = M1_SIZE * sizeof(float);
    float h_m1[M1_SIZE];
    for (int i = 0; i < M1_SIZE; i++) 
    {
        h_m1[i] = float(i); // 0, 1, .. 11
    }
    float h_out[M1_SIZE];   // sigmoid

    // GPU
    float *d_m1;
    float *d_out;
    cudaMalloc((void**) &d_m1, M1_BYTES);
    cudaMalloc((void**) &d_out, M1_BYTES);

    // sigmoid
    cudaMemcpy(d_m1, h_m1, M1_BYTES, cudaMemcpyHostToDevice);
    dSigmoid(d_m1, d_out, 4, 3);
    cudaMemcpy(h_out, d_out, M1_BYTES, cudaMemcpyDeviceToHost);
    // print result
    printf("sigmoid\n");
    for (int i = 0; i < M1_BYTES; i++)
    {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // sigmoid derivative
    cudaMemcpy(d_m1, h_out, M1_BYTES, cudaMemcpyHostToDevice);
    dSigmoid_d(d_m1, d_out, 4, 3);
    cudaMemcpy(h_out, d_out, M1_BYTES, cudaMemcpyDeviceToHost);
    // print result
    printf("sigmoid derivative\n");
    for (int i = 0; i < M1_BYTES; i++)
    {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // free memory
    cudaFree(d_m1);
    cudaFree(d_out);
    // free(h_m1);
    // free(h_m2);
    // free(h_out);

}