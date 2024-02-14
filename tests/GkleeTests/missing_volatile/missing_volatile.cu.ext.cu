#include <cstdio>

#define N 32

__global__ void k(volatile int* in)
{
  __shared__ int volatile smem[N];
  __shared__ int volatile tmem[N];


  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  smem[idx] = in[idx];
  tmem[idx] = smem[N-idx-1];
  
  in[idx] = tmem[idx];
}

