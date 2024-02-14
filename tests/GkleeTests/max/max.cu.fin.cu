#include <cstdio>
#include <cstdlib>

#define N 8

__global__ void mmax(int* in, int* out)
{
  // ask Mark, I have no idea - Ian
  int idx = threadIdx.x + blockDim.x*blockIdx.x;

  __shared__ int rslt[N];
  rslt[idx] = in[idx];
  
  int lim = N/2;
  int temp = 0;
  while (lim > 0) {
    __syncthreads();
    if(idx < lim) {
      temp = max(rslt[2*idx], rslt[2*idx + 1]);
    }
 
    __syncthreads();
    rslt[idx] = temp;
    lim /= 2;
  }
  if (idx == 0)
    *out = rslt[0];
}

