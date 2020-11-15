CUDA_LIBS=-L/usr/local/cuda/lib64 -lcudadevrt  -L/usr/lib/aarch64-linux-gnu/ -lcublas_device
#CUDA_LIB=-L/usr/local/cuda/lib64 -lcudadevrt  -L/usr/lib/x86_64-linux-gnu/ -lcublas_device
CUDA_PATH ?= "/usr/local/cuda"
NVCC := $(CUDA_PATH)/bin/nvcc
NVCC_OPTS=-O3 -arch=sm_53 -Xcompiler -Wall -Xcompiler -Wextra -m64 -G -g -ccbin aarch64-linux-gnu-g++ -rdc=true 

all: one_layer_perceptron

one_layer_perceptron: one_layer_perceptron.cu Makefile
	$(NVCC) -o one_layer_perceptron one_layer_perceptron.cu $(CUDA_LIBS) $(NVCC_OPTS)
	
clean:
	rm -f *.o one_layer_perceptron