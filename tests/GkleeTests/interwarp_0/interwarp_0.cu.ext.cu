#include <cstdio>

__global__ void iwarp(int* out)
{
  volatile int* vout = out;
  *vout = threadIdx.x;
}

