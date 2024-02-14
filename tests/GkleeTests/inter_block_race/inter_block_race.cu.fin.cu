#define N 128
#define B 2

__global__ void k(int* in)
{
  in[threadIdx.x] = blockIdx.x;
}

