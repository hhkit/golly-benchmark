#include <stdio.h>

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void device_global(unsigned int *input_array, int num_elements, int shift_by) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  my_index += shift_by;
  for (int i=0; i<my_index; i++) {
    if (input_array[i] != 0) {
      if ((my_index+2) % input_array[i] == 0) {
	input_array[my_index] = 0;
      }
    }
  }
}


