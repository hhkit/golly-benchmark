#include <stdio.h>
#include <cuda.h>
// from http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }



__global__ void test()
{
  __shared__ int i, mutex;

  if (threadIdx.x == 0) {
    i = 0;
    mutex = 0;
  }
  __syncthreads();

  while( atomicCAS(&mutex, 0, 1) != 0);
  i++;
  printf("thread %d: %d\n", threadIdx.x, i);
  atomicExch(&mutex,0);
}


