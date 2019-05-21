all: error.cu
	nvcc -arch=sm_35 error.cu -o error

debug: error.cu
	nvcc -g -G -arch=sm_35 error.cu -o error
