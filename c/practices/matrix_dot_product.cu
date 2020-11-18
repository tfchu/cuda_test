// $ nvcc -o matrix_dot_product matrix_dot_product.cu

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// kernel of product of two matrices: m1 x m2
__global__ 
void kDot(const int nThreads, const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){
	/*  Computes the product of two matrices: m1 x m2.
	   	Inputs:
	    m1: array, left matrix of size m1_rows x m1_columns
	    m2: array, right matrix of size m1_columns x m2_columns (the number of rows in the right matrix
	    must be equal to the number of the columns in the left one)
	    output: array, the results of the computation are to be stored here:
	    		m1 * m2, product of two arrays m1 and m2, a matrix of size m1_rows x m2_columns
	    m1_rows: int, number of rows in the left matrix m1
	    m1_columns: int, number of columns in the left matrix m1
	    m2_columns: int, number of columns in the right matrix m2
	*/

	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		 i < nThreads;
		 i += blockDim.x * gridDim.x)
	{
	    int r = (int)i / m2_columns;		// 
	    int c = i % m2_columns;
	    float t_output = 0.f;

	    for( int k = 0; k < m1_columns; ++k ) {
	        t_output += m1[ r * m1_columns + k ] * m2[ k * m2_columns + c ];
	    }

	    output[i] = t_output;
	}
}

// product of two matrices: m1 x m2
// output is m1_rows x m2_columns
// __device__ 
void dDot(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){

	kDot <<< m1_rows, m2_columns >>> (m1_rows * m2_columns, m1, m2, output, m1_rows , m1_columns, m2_columns );
	cudaDeviceSynchronize();
	//return output;
}

int main(void)
{
    // host initialization
    const int M1_SIZE = 9;      // 3x3 matrix
    const int M1_BYTES = M1_SIZE * sizeof(float);
    const int M2_SIZE = 6;      // 3x2 matrix
    const int M2_BYTES = M2_SIZE * sizeof(float);
    const int PRODUCT_SIZE = 6;
    const int PRODUCT_BYTES = PRODUCT_SIZE * sizeof(float);
    float h_m1[M1_SIZE];
    for (int i = 0; i < M1_SIZE; i++) 
    {
        h_m1[i] = float(i); // 0, 1, .. 9
    }
    float h_m2[M2_SIZE] = {3., 4., 5., 6., 7., 8.};
    float h_out[PRODUCT_SIZE];

    // GPU
    float *d_m1, *d_m2;
    float *d_out;
    cudaMalloc((void**) &d_m1, M1_BYTES);
    cudaMalloc((void**) &d_m2, M2_BYTES);
    cudaMalloc((void**) &d_out, PRODUCT_BYTES);
    cudaMemcpy(d_m1, h_m1, M1_BYTES, cudaMemcpyHostToDevice);
    dDot(d_m1, d_m2, d_out, 3, 3, 2);
    cudaMemcpy(h_out, d_out, PRODUCT_BYTES, cudaMemcpyDeviceToHost);

    // print result
    for (int i = 0; i < PRODUCT_SIZE; i++)
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