#include <stdio.h>
#define THREADS 32

// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void device_global(unsigned int *input_array, int num_elements) {
  int my_index = blockIdx.x * blockDim.x + threadIdx.x;

  // all threads write their index into the array
  input_array[my_index] = my_index;

  __syncthreads();

  // all threads write to the array from values in th array
  // written by the second neighbor thread, causing a read-write race
  // the mod is so the last even thread reads from the first
  // even thread's index
  //
  // since the value never goes into an indexing position there is no
  // flow
  input_array[my_index] = input_array[(my_index+2) % THREADS];
}


