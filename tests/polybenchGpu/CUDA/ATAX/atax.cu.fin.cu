/**
 * atax.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda.h>

#define POLYBENCH_TIME 1

#include "atax.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0


#ifndef M_PI
#define M_PI 3.14159
#endif

#define RUN_ON_CPU











__global__ void atax_kernel1(int nx, int ny, DATA_TYPE *A, DATA_TYPE *x, DATA_TYPE *tmp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < _PB_NX)
	{
		tmp[i] = 0;
		int j;
		for(j=0; j < _PB_NY; j++)
		{
			tmp[i] += A[i*NY+j] * x[j];
		}
	}
}

__global__ void atax_kernel2(int nx, int ny, DATA_TYPE *A, DATA_TYPE *y, DATA_TYPE *tmp)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (j < _PB_NY)
	{
		y[j] = 0;
		int i;
		for(i=0; i < _PB_NX; i++)
		{
			y[j] += A[i*NY+j] * tmp[i];
		}
	}
}








/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */





#include "../../common/polybench.c"
