/*
compile 
$ nvcc -o matrix_transpose_dot_product matrix_transpose_dot_product.cu

elementwise multiplication and subtraction

numpy version 
import numpy as np
m1 = np.array(((0, 1, 2), (3, 4, 5), (6, 7, 8)))
m2 = np.array(((8, 7, 6), (5, 4, 3), (2, 1, 0)))
m1.dot(m2.T)    # m1 dot m2_transpose (m1_m2T)
m1.T.dot(m2)    # m1_transpose dot m2 (m1T_m2)
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

__global__ 
void kDot_m1_m2T(const int nThreads, const float *m1, const float *m2, float *output, const int m1_columns, const int m2_rows ){
	/*  Updates the output matrix with the product of two matrices: m1 and m2 transposed.
	   	Inputs:
	    m1: array, left matrix of size m1_rows x m1_columns
	    m2: array, right matrix of size m2_rows x m1_columns (m2 transposed will be of size m1_columns x m2_rows)
	    output: array, the results of the computation are to be stored here:
	    		m1 * m2, product of two arrays m1 and m2, a matrix of size m1_rows x m2_rows
	    m1_columns: int, number of columns in the left matrix m1
	    m2_rows: int, number of rows in the left matrix m2
	*/

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
		int r = (int)i / m2_rows;
		int c = i % m2_rows;
		float t_output = 0.0;
		int id_T;

		for( int k = 0; k < m1_columns; ++k ) {
			id_T = c * m1_columns + k;
			t_output += m1[ r * m1_columns + k ] * m2[ id_T ];
		}

		output[i] = t_output;
	}
}

float* dDot_m1_m2T(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_rows )
{
	kDot_m1_m2T <<< m1_rows, m2_rows >>> ( m1_rows * m2_rows, m1, m2, output, m1_columns, m2_rows );
	cudaDeviceSynchronize();
	return output;
}

__global__ 
void kDot_m1T_m2(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows,
							const int m1_columns, const int m2_columns ){
	/*  Increments the output matrix with the product of two matrices: m1 transposed and m2.
	   	Inputs:
	    m1: array, left matrix of size m1_rows x m1_columns (m1 transposed will be of size m1_columns x m1_rows)
	    m2: array, right matrix of size m1_rows x m2_columns
	    output: array, the results of the computation are to be stored here:
	    		m1 * m2, product of two arrays m1 and m2, a matrix of size m1_columns x m2_columns
	    m1_rows: int, number of rows in the left matrix m1
	    m1_columns: int, number of columns in the left matrix m1
	    m2_rows: int, number of rows in the left matrix m2
	*/

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
	    int r = (int)i / m2_columns;
	    int c = i % m2_columns;
	    int id_T;
	    float t_output = 0.0;

	    for( int k = 0; k < m1_rows; ++k ) {
	    	id_T = k * m1_columns + r;
	        t_output += m1[ id_T ] * m2[ k * m2_columns + c ];
	    }

	    output[i] += t_output;
	}
}

void dDot_m1T_m2(const float *m1, const float *m2, float *output, const int m1_height , const int m1_width, const int m2_width )
{
	kDot_m1T_m2 <<< m1_width, m2_width >>> (m1_width * m2_width, m1, m2, output, m1_height, m1_width, m2_width );
	cudaDeviceSynchronize();
}

int main(void)
{
    // host initialization
    const int M_SIZE = 9;          // 3x3 matrix
    const int M_BYTES = M_SIZE * sizeof(float);
    float h_m1[M_SIZE], h_m2[M_SIZE], h_out[M_SIZE];
    for (int i = 0; i < M_SIZE; i++) 
    {
        h_m1[i] = float(i);                 // 0, 1, 2, 3, 4, 5, 6, 7, 8
        h_m2[i] = float(M_SIZE - 1 - i);    // 8, 7, 6, 5, 4, 3, 2, 1, 0
    }

    // GPU memory allocation and initialization
    float *d_m1, *d_m2, *d_out;
    cudaMalloc((void**) &d_m1, M_BYTES);
    cudaMalloc((void**) &d_m2, M_BYTES);
    cudaMalloc((void**) &d_out, M_BYTES);
    cudaMemcpy(d_m1, h_m1, M_BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, M_BYTES, cudaMemcpyHostToDevice);

    // m1_transpose dot m2
    dDot_m1T_m2(d_m1, d_m2, d_out, 3, 3, 3);
    cudaMemcpy(h_out, d_out, M_BYTES, cudaMemcpyDeviceToHost);
    // print result
    printf("m1_transpose dot m2\n");
    for (int i = 0; i < M_SIZE; i++)
    {
        printf("h_out[%d] = %f\n", i, h_out[i]);
    }

    // m1 dot m2_transpose
    dDot_m1_m2T(d_m1, d_m2, d_out, 3, 3, 3);
    cudaMemcpy(h_out, d_out, M_BYTES, cudaMemcpyDeviceToHost);
    // print result
    printf("m1 dot m2_transpose\n");
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