// Tests executing two kernels, with host code between kernel launches.

#include <cstdio>

#define N 100

__global__ void kernel1(int* in, int* out)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < N)
    out[idx] = in[idx] + 1;
}

__global__ void kernel2(int*in, int*out)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if(idx < N)
    out[idx] = in[idx]*in[idx];
}

