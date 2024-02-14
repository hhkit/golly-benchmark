#include <cstdlib>
#include <time.h>

#define DIM1 3
#define DIM2 3

__global__ void avg(float* in, float* out, int radius)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if(tid < DIM1 * DIM2)
    {
      int x = tid / DIM1;
      int y = tid % DIM2;
      
      float count = 0;
      float val = 0;

      for(int i = -1 * radius; i <= radius; i++)
	{
	  int nx = i + x;
	  if(nx >= 0 && nx < DIM1)
	    for(int j = -1 * radius; j <= radius; j++)
	      {
		int ny = j + y;
		if(i*i + j*j <= radius * radius && ny >= 0 && ny < DIM2)
		  val += in[nx * DIM1 + ny], count++;
	      }
	}
      out[tid] = val/count;
    }  
}


