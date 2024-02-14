#define N 16

__global__ void k(int* in)
{
  if(threadIdx.x < N)
    in[0] = threadIdx.x;
}

