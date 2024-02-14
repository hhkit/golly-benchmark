#include <stdio.h>

__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;
  int index = my_index%num_elements;
  input_array[index] = my_index;
}

