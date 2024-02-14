#include <stdio.h>

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  // stop out of bounds access
  if (my_index < num_elements) {

    if (my_index%2 == 1) {
      // even threads write their index to their array entry
      input_array[my_index] = my_index;
    } else {
      // odd threads copy their value from the next array entry
      input_array[my_index] = input_array[my_index+1];
    }
  }
}

