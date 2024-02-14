#include <cstdio>

#define N 64
#define B 1
#define T 64

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }


__global__ void dl(int* in)
{
  
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  // The warps in this block take different paths; the synctreads calls
  // will cause a deadlock.
  if(tid > 31)
  {
    if(in[tid] % 2 == 0)
      in[tid]++;

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

