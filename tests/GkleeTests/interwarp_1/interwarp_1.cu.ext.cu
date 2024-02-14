#include <cstdio>

#define N 32

__global__ void iwarp(int* out)
{
  __shared__ volatile int smem[32];
  volatile int* vout = out;
  int idx = threadIdx.x;
  smem[idx] = vout[idx];

  if(idx % 2 == 0)
    smem[idx] = 1;
  else
    smem[idx-1] = 0;
  vout[idx] = smem[idx];
}

