#include <cstdio>

#define N 50
#define T 128
#define B 2

__global__ void div(int* in, int* out)
{
  int tid = threadIdx.x + blockDim.x*blockIdx.x;
  if(tid < N)
  {
    if(tid % 2 == 0)
      out[tid] = in[tid] - 1;
    else
      out[tid] = in[tid] + 1;
  }
}

