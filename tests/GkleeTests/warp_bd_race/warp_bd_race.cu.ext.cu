#define N 16

__global__ void k(int* in)
{
  if(threadIdx.x < N)
    if(threadIdx.x % 2 == 0)
      in[0] = 0;
    else
      in[0] = 1;
}

