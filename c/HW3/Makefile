NVCC=/usr/local/cuda/bin/nvcc
GCC=/usr/bin/aarch64-linux-gnu-g++

###################################
# These are the default install   #
# locations on most linux distros #
###################################

#OPENCV_LIBPATH=/usr/lib
#OPENCV_INCLUDEPATH=/usr/include

###################################################
# On Macs the default install locations are below #
###################################################

#OPENCV_LIBPATH=/usr/local/lib
#OPENCV_INCLUDEPATH=/usr/local/include

#OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui

#CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include

######################################################
# On Macs the default install locations are below    #
# ####################################################

#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib

######################################################
# Jetson nano                                        #
# ####################################################
# OPENCV_LIBPATH=/usr/lib/aarch64-linux-gnu
# OPENCV_INCLUDEPATH=/usr/include/opencv4
# OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs	# link required libs (otherwise "undefined reference")
# CUDA_INCLUDEPATH=/usr/local/cuda/include
# CUDA_LIBPATH=/usr/local/cuda/lib64
# NVCC_OPTS=-O3 -arch=sm_53 -Xcompiler -Wall -Xcompiler -Wextra -m64 -G -g
# GCC_OPTS=-O3 -Wall -Wextra -g

######################################################

# Ubuntu 16.04.3                                     #

# ####################################################
#OPENCV_LIBPATH=/usr/lib/x86_64-linux-gnu
#OPENCV_INCLUDEPATH=/usr/include
#OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
#CUDA_INCLUDEPATH=/usr/local/cuda/include
#CUDA_LIBPATH=/usr/local/cuda/lib64
#NVCC_OPTS=-O3 -arch=sm_53 -Xcompiler -Wall -Xcompiler -Wextra -m64 -G -g -ccbin aarch64-linux-gnu-g++
#GCC_OPTS=-O3 -Wall -Wextra -g

######################################################

# Ubuntu 18.04.5 opencv aarch64                      #

# ####################################################
OPENCV_INCLUDEPATH=/home/test/opencv-4.1.1/build/install/include/opencv4
OPENCV_LIBPATH=-L/home/test/opencv-4.1.1/build/install/lib
OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
CUDA_INCLUDEPATH=/usr/local/cuda/include
CUDA_LIBPATH=/usr/local/cuda/lib64
NVCC_OPTS=-O3 -arch=sm_53 -Xcompiler -Wall -Xcompiler -Wextra -m64 -G -g -ccbin aarch64-linux-gnu-g++
GCC_OPTS=-O3 -Wall -Wextra -g

#NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64

#GCC_OPTS=-O3 -Wall -Wextra -m64

all: student 

student: main.o student_func.o HW3.o loadSaveImage.o compare.o reference_calc.o Makefile
	$(NVCC) -o HW3 main.o student_func.o HW3.o loadSaveImage.o compare.o reference_calc.o $(OPENCV_LIBPATH) $(OPENCV_LIBS) $(NVCC_OPTS)

main.o: main.cpp timer.h utils.h reference_calc.h compare.h
	$(GCC) -c main.cpp $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

HW3.o: HW3.cu loadSaveImage.h utils.h
	$(NVCC) -c HW3.cu -I $(OPENCV_INCLUDEPATH) $(NVCC_OPTS)

loadSaveImage.o: loadSaveImage.cpp loadSaveImage.h
	$(GCC) -c loadSaveImage.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

compare.o: compare.cpp compare.h
	$(GCC) -c compare.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

reference_calc.o: reference_calc.cpp reference_calc.h
	$(GCC) -c reference_calc.cpp -I $(OPENCV_INCLUDEPATH) $(GCC_OPTS) -I $(CUDA_INCLUDEPATH)

student_func.o: student_func.cu utils.h
	$(NVCC) -c student_func.cu $(NVCC_OPTS)

clean:
	rm -f *.o hw
	find . -type f -name '*.exr' | grep -v memorial | xargs rm -f