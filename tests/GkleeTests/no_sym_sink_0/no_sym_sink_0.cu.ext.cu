#include <cstdio>

#define N 8

__global__ void k(int* in)
{
  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  // Gkleep reports out of bound pointer...
  // Causes gkleep to use one flow per thread.
  int write_val = N-tid-1;

  for(int i = 0; i <= tid; i++)
    in[tid] = write_val;
  
  __syncthreads();

  if(tid % 2 == 1)
    in[tid] = in[in[tid]];  
}

