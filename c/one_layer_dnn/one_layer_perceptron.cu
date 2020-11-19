/* 
onehiddenlayerperceptron.cu
onehiddenlayerperceptron

Created by Sergei Bugrov on 8/21/17.
Copyright © 2017 Sergei Bugrov. All rights reserved.

output
$ ./one_layer_perceptron
Prediction[0] : 0.060997 True Value[0] : 0.000000 Error[0] : 0.060997
Prediction[1] : 0.076193 True Value[1] : 0.000000 Error[1] : 0.076193
Prediction[2] : 0.927551 True Value[2] : 1.000000 Error[2] : -0.072449
Prediction[3] : 0.918263 True Value[3] : 1.000000 Error[3] : -0.081737
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
__device__ 
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
__device__ 
float* dMatrixSubstractMatrix(const float *m1, const float *m2, float *output, const int width, const int height){

	kMatrixSubstractMatrix <<< width, height >>> ( width * height, m1, m2, output );
    cudaDeviceSynchronize();
    return output;
}

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
__device__ 
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
__device__ 
float* dSigmoid_d(float const *input, float *output, const int rows, const int columns){
	kSigmoid_d <<< rows, columns >>> (rows*columns, input, output);
	cudaDeviceSynchronize();
	return output;
}

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
__device__ 
float* dDot(const float *m1, const float *m2, float *output, const int m1_rows , const int m1_columns, const int m2_columns ){

	kDot <<< m1_rows, m2_columns >>> (m1_rows * m2_columns, m1, m2, output, m1_rows , m1_columns, m2_columns );
	cudaDeviceSynchronize();
	return output;
}

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

__device__ 
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

__device__ 
void dDot_m1T_m2(const float *m1, const float *m2, float *output, const int m1_height , const int m1_width, const int m2_width )
{
	kDot_m1T_m2 <<< m1_width, m2_width >>> (m1_width * m2_width, m1, m2, output, m1_height, m1_width, m2_width );
	cudaDeviceSynchronize();
}

__device__ 
void kPrintMatrix (const float* M, int h, int w) {
    /*  Prints out the input array as h x w matrix.
     Inputs:
     m: vector, matrix of size n_rows x n_columns
     h: int, number of rows in the matrix M
     w: int, number of columns in the matrix M
     */
	for (int i = 0; i < h; i++){
		for (int j = 0; j < w; j++){
			printf("%f  ", M[i*w+j]);
		}
		printf("\n");
	}
	printf("\n");
}

// whole training process
// X (d_X): pointer to training data, TRAINING_SIZE x TRAINING_DIM = 4 x 4
// X_w: training data width, or number of training data (TRAINING_DIM)
// X_h: training data height, or number of features (TRAINING_SIZE)
// y: pointer to labels of training data
// y_w: 1
// l1: pointer to Layer 1 output?, L1_SIZE x TRAINING_SIZE
// l1_w: layer 1 width, or L1_SIZE (8)
// l_1_d: pointer to layer 1 delta
// pred: pointer to prediction, e.g. pred[0] is prediction of 1st data
// pred_d: prediction delta
// W0: weight 0
// W1: weight 1
// buffer
/*
input X wiht 4 features -> W0 (8x4 matrix) -> hidden layer 1 with 8 neurons -> sigmoid -> W1 (1x8 matrix) -> output
*/
__global__ 
void kFit(	const float* X, const int X_w, const int X_h,
						const float* y, const int y_w,
						float* l1, const int l1_w, float* l_1_d,
						float* pred, float* pred_d,
						float* W0,
						float* W1,
						float* buffer
						)
{
	// print
	printf("W1\n");
	kPrintMatrix(W1, 1, 8);
	printf("W0\n");
	kPrintMatrix(W0, 8, 4);
	printf("\n");
	
	// for 50 iterations
	for (unsigned i = 0; i < 50; ++i) {
		// forward propagation
		// activate(input * weight) for input > layer 1 and for layer 1 > output
		/*
		line 1: 
			X = input, l1 = (X dot W0) = in_h1 (h1: hidden layer 1 )
			l1 = sigmoid(l1) = out_h1 (1 x 8)
		line 2: 
			pred = (1 dot W1) = IN_output, pred = sigmoid(pred) = output
		*/
        dSigmoid(dDot(X, W0, l1, X_h, X_w, l1_w), l1, X_h, l1_w);		
		dSigmoid(dDot(l1, W1, pred, X_h, l1_w, y_w), pred, X_h, y_w);
		
		// backpropagate errors
		/* 
		line 1: 
			x or *: elementwise matrix multiplication; dot: matrix dot product
			pred_d = y - pred (my: y_cap - y)
			buffer = sigmoid_d(pred)
			pred_d = pred_d x buffer = (y - y_cap) * y * (1 - y)
		line 2: 
			l_1_d = (pred_d dot W1_transpose)
			buffer = sigmoid_d(l1)
			l_1_d = l_1_d x buffer
		*/
        dMatrixByMatrixElementwise(dMatrixSubstractMatrix(y, pred, pred_d, X_h, y_w), dSigmoid_d(pred, buffer, X_h, y_w), pred_d, X_h, y_w );
		dMatrixByMatrixElementwise(dDot_m1_m2T(pred_d, W1, l_1_d, X_h, y_w, l1_w), dSigmoid_d(l1, buffer, X_h, l1_w), l_1_d, X_h, l1_w);

		// update weights 0 and 1
		/*
		line 1: W1 = (l1_transpose dot pred_d)
		line 2: W0 = (X_transpose dot l_1_d)
		*/
        dDot_m1T_m2( l1, pred_d, W1, X_h, l1_w, y_w );
		dDot_m1T_m2( X, l_1_d, W0, X_h, X_w, l1_w );
		// print
		printf("W1\n");
		kPrintMatrix(W1, 1, 8);
		printf("W0\n");
		kPrintMatrix(W0, 8, 4);
		printf("\n");
    }
}

int main(void){
	// h_: host (CPU)
	// d_: device (GPU)

	// Define hyperparameters
	const int TRAINING_SIZE = 4;	// number of samples
	const int TRAINING_DIM = 4;		// number of features
	const int L1_SIZE = 8;

	// X, the first 4 lines from Iris dataset
	// sepal length, sepal width, petal length, petal width
	// 4 points with 4 features each 
	float h_X[TRAINING_SIZE*TRAINING_DIM] = {	5.1, 3.5, 1.4, 0.2,
												4.9, 3.0, 1.4, 0.2,
												6.2, 3.4, 5.4, 2.3,
												5.9, 3.0, 5.1, 1.8 };
	
	// define input size
	const signed int X_size = sizeof(h_X);
	
	// allocate space on GPU for input data
	float *d_X;
	cudaMalloc(&d_X, X_size);
	cudaMemcpy(d_X, h_X, X_size, cudaMemcpyHostToDevice);

	// initialize weight 0 randomly
	const long signed int W0_size = L1_SIZE*TRAINING_DIM*sizeof(float);
	float *h_W0 = (float*)malloc(W0_size);
	for (int i = 0; i < L1_SIZE*TRAINING_DIM; i++){
	    h_W0[i] = 0.1 * (2.0*rand()/RAND_MAX-1.0);
	}

	// allocate space on GPU for weight 0
	float *d_W0;
	cudaMalloc(&d_W0, W0_size);
	cudaMemcpy(d_W0, h_W0, W0_size, cudaMemcpyHostToDevice);

	// layer 1 size
	// LAYER_1, LAYER_1_DELTA AND BUFFER OF LAYER 1 SIZE
	const long signed int L1_size = L1_SIZE*TRAINING_SIZE*sizeof(float);

	// define layer 1 variables (delta and buffer)
	float* h_layer_1 = (float*)malloc(L1_size);
	float* h_layer_1_delta = (float*)malloc(L1_size);
	float* h_buffer = (float*)malloc(L1_size);

	// give layer 1 initial 0 values
	for (int i = 0; i < L1_SIZE*TRAINING_SIZE; i++){
	    h_layer_1[i] = 0.0;
	    h_buffer[i] = 0.0;
	    h_layer_1_delta[i] = 0.0;
	}

	// allocate GPU space for layer 1
	float *d_layer_1;
	cudaMalloc(&d_layer_1, L1_size);
	cudaMemcpy(d_layer_1, h_layer_1, L1_size, cudaMemcpyHostToDevice);

	// allocate GPU space for layer 1 buffer 
	float *d_buffer;
	cudaMalloc(&d_buffer, L1_size);
	cudaMemcpy(d_buffer, h_buffer, L1_size, cudaMemcpyHostToDevice);

	// allocate GPU space for layer 1 data
	float *d_layer_1_delta;
	cudaMalloc(&d_layer_1_delta, L1_size);
	cudaMemcpy(d_layer_1_delta, h_layer_1_delta, L1_size, cudaMemcpyHostToDevice);

	// allocate weight 1 randomly
	const long signed int W1_size = L1_SIZE*sizeof(float);
	float *h_W1 = (float*)malloc(W1_size);
	for (int i = 0; i < L1_SIZE; i++){
	    h_W1[i] = 0.1* (2.0*rand()/RAND_MAX-1.0);
	}

	// allocate GPU space for weight 1
	float *d_W1;
	cudaMalloc(&d_W1, W1_size);
	cudaMemcpy(d_W1, h_W1, W1_size, cudaMemcpyHostToDevice);

	// 4 labels
	float h_y[4] = {	0,
						0,
						1,
						1 };

	// allocate GPU space for labels 
	const signed int y_size = sizeof(h_y);
	float *d_y;
	cudaMalloc(&d_y, y_size);
	cudaMemcpy(d_y, h_y, y_size, cudaMemcpyHostToDevice);

	// define prediction and prediction delta
	float* h_pred = (float*)malloc(y_size);
	float* h_pred_delta = (float*)malloc(y_size);
	for (int i = 0; i < TRAINING_SIZE; i++){
	    h_pred[i] = 0.0;
	    h_pred_delta[i] = 0.0;
	}

	// allocate GPU space for predciction
	float *d_pred;
	cudaMalloc(&d_pred, y_size);
	cudaMemcpy(d_pred, h_pred, y_size, cudaMemcpyHostToDevice);

	// allocate GPU space for prediction delta
	float *d_pred_delta;
	cudaMalloc(&d_pred_delta, y_size);
	cudaMemcpy(d_pred_delta, h_pred_delta, y_size, cudaMemcpyHostToDevice);

	// train model on 1 block with 1 thread given initial vars
	kFit <<< 1, 1 >>> (	d_X, TRAINING_DIM, TRAINING_SIZE,
						d_y, 1,
						d_layer_1, L1_SIZE, d_layer_1_delta,
						d_pred,
						d_pred_delta,
						d_W0,
						d_W1,
						d_buffer);
	
	// send computed vars back to host
	cudaMemcpy(h_pred, d_pred, y_size, cudaMemcpyDeviceToHost);

	// free GPU memory
	cudaFree(d_pred);
	cudaFree(d_X);
	cudaFree(d_y);
	cudaFree(d_layer_1_delta);
	cudaFree(d_pred_delta);
	cudaFree(d_W0);
	cudaFree(d_W1);
	cudaFree(d_buffer);

	// free CPU memory
	free(h_layer_1_delta);
	free(h_pred_delta);
	free(h_W0);
	free(h_W1);
	free(h_buffer);

	// print prediction, label and error values
	for (int i = 0; i < TRAINING_SIZE; i++){
		printf("Prediction[%i] : %f True Value[%i] : %f Error[%i] : %f\n", i, h_pred[i], i, h_y[i], i, h_pred[i] - h_y[i]);
	}

	// free CPU memory of final prediction var
	free(h_pred);
}