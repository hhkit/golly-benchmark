#include <stdio.h>
#define THREADS 32

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i=0; i<THREADS; i++) {
    if (i<my_index) {
      if (((my_index+2) % input_array[i]) == 0) {
	input_array[my_index] = 0;
      }
    }
    __syncthreads();
  }
}


