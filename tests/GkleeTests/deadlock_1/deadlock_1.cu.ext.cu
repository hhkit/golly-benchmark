#include <cstdio>

#define N 64
#define B 2
#define T 32

__global__ void dl(int* in)
{
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(blockIdx.x % 2 == 0)
  {
    if(in[tid] % 2 == 0)
      in[tid]++;
    // Fine because conditional synchronization will
    // happen within a block.
    __syncthreads();

  }
  else {
    if(in[tid] % 2 == 1)
      in[tid]--;
    
    __syncthreads();
  }
/*  int sum = in[tid];
  if(tid > 0)
    sum += in[tid-1];
  if(tid < N - 1)
      sum += in[tid+1];
      in[tid] = sum / 3; */
}

