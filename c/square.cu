#include <stdio.h>

__global__ 
void square(float * out, float * in){
    int idx = threadIdx.x;
    float f = in[idx];
    out[idx] = f * f;
}

int main(int argc, char ** argv) {
	const int ARRAY_SIZE = 64;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

	// declare memory pointers
	float * in, * out;

	// Allocate Unified Memory – accessible from CPU or GPU
	cudaMallocManaged(&in, ARRAY_BYTES);
	cudaMallocManaged(&out, ARRAY_BYTES);

	// generate the input array
	for (int i = 0; i < ARRAY_SIZE; i++) {
		in[i] = float(i);
	}

	// launch the kernel
    square<<<1, ARRAY_SIZE>>>(out, in);
	//square<<<dim3(2,1,1), dim3(32,1,1)>>>(out, in);

	// Wait for GPU to finish before accessing on host
	cudaDeviceSynchronize();

	// print out the resulting array
	for (int i =0; i < ARRAY_SIZE; i++) {
		printf("%f", out[i]);
		printf(((i % 4) != 3) ? "\t" : "\n");
	}

	cudaFree(in);
	cudaFree(out);

	return 0;
}