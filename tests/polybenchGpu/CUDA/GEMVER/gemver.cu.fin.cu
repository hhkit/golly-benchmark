/**
 * gemver.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "gemver.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0


#define RUN_ON_CPU














__global__ void gemver_kernel1(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *v1, DATA_TYPE *v2, DATA_TYPE *u1, DATA_TYPE *u2)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < _PB_N) && (j < _PB_N))
	{
		a[i * N + j] += u1[i] * v1[j] + u2[i] * v2[j];
	}
}


__global__ void gemver_kernel2(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_N)
	{
		int j;
		for(j = 0; j < _PB_N; j++) 
		{
			x[i] += beta * a[j * N + i] * y[j];
		}
		x[i] += z[i];
	}
}


__global__ void gemver_kernel3(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *a, DATA_TYPE *x, DATA_TYPE *w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i >= 0) && (i < _PB_N))
	{
		int j;
		for(j = 0; j < _PB_N; j++)
		{ 
			w[i] += alpha * a[i*N + j] * x[j];
		}
	}
}





/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */

	



#include "../../common/polybench.c"
