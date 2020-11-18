/*
compile 
$ nvcc -o matrix_elementwise matrix_elementwise.cu

elementwise multiplication and subtraction

numpy version 
import numpy as np
m1 = np.array(((0, 1, 2), (3, 4, 5), (6, 7, 8)))
m2 = np.array(((8, 7, 6), (5, 4, 3), (2, 1, 0)))
m1*m2       # or np.multiply(m1, m2)
m1-m2
*/
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// kernel of elementwise multiplication of 2 arrays
__global__ 
void kMatrixByMatrixElementwise(const int nThreads, const float *m1, const float *m2, float *output) {
    /*  Computes the product of two arrays (elementwise multiplication).
     Inputs:
     m1: array
     m2: array
     output: array,the results of the multiplication are to be stored here
    */
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = m1[i] * m2[i];
	  }
}

// elementwise multiplication of 2 arrays
float* dMatrixByMatrixElementwise(const float *m1, const float *m2, float *output, const int width, const int height){

	kMatrixByMatrixElementwise <<< width, height >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

// kernel elementwise difference of 2 arrays
__global__ 
void kMatrixSubstractMatrix(const int nThreads, const float *m1, const float *m2, float *output) {
    /*  Computes the (elementwise) difference between two arrays
     Inputs:
     m1: array
     m2: array
     output: array,the results of the computation are to be stored here
     */

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	  {
		output[i] = m1[i] - m2[i];
	  }
}

// elementwise difference of 2 arrays
float* dMatrixSubstractMatrix(const float *m1, const float *m2, float *output, const int width, const int height){

	kMatrixSubstractMatrix <<< width, height >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

int main(void)
{
    // host initialization
    const int M_SIZE = 9;          // 3x3 matrix
    const int M_BYTES = M_SIZE * sizeof(float);
    float h_m1[M_SIZE], h_m2[M_SIZE], h_out[M_SIZE];
    for (int i = 0; i < M_SIZE; i++) 
    {
        h_m1[i] = float(i); // 0, 1, .. 8
        h_m2[i] = float(M_SIZE - 1 - i);
    }

    // GPU memory allocation and initialization
    float *d_m1, *d_m2, *d_out;
    cudaMalloc((void**) &d_m1, M_BYTES);
    cudaMalloc((void**) &d_m2, M_BYTES);
    cudaMalloc((void**) &d_out, M_BYTES);
    cudaMemcpy(d_m1, h_m1, M_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, M_BYTES, cudaMemcpyHostToDevice);

    // elementwise subtraction
    dMatrixSubstractMatrix(d_m1, d_m2, d_out, 3, 3);
    cudaMemcpy(h_out, d_out, M_BYTES, cudaMemcpyDeviceToHost);
    // print result
    printf("elementwise subtraction\n");
    for (int i = 0; i < M_SIZE; i++)
    {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // elementwise multiplication
    dMatrixByMatrixElementwise(d_m1, d_m2, d_out, 3, 3);
    cudaMemcpy(h_out, d_out, M_BYTES, cudaMemcpyDeviceToHost);
    // print result
    printf("elementwise multiplication\n");
    for (int i = 0; i < M_SIZE; i++)
    {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // free memory
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_out);
    // free(h_m1);
    // free(h_m2);
    // free(h_out);

}