#include <stdio.h>
#define THREADS 64

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;

  // all threads write a value to the array
  input_array[my_index] = my_index - (my_index%2);

  __syncthreads(); // all initial values are written

  // all threads grab a value from the array
  // we know this will always be in bounds
  int new_index = input_array[my_index];
  
  __syncthreads(); // all values are read

  // use the values to write to the array, a write-write race
  input_array[new_index] = my_index;
}


