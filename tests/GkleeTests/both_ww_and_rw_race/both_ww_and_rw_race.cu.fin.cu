// Exhibits a data race (RW, WW) in global memory.
// Gklee and Gkleep both detect.


#define N 50
#define T 128
#define B 2

__global__ void colonel(int* in)
{
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  in[tidx%N]++;
}

