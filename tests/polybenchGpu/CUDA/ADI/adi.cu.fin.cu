/**
 * adi.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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

#define POLYBENCH_TIME 1

#include "adi.cuh"
#include "../../common/polybench.h"
#include "../../common/polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 2.5

#define GPU_DEVICE 0

#define RUN_ON_CPU














__global__ void adi_kernel1(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < _PB_N))
	{
		for (int i2 = 1; i2 < _PB_N; i2++)
		{
			X[i1*N + i2] = X[i1*N + i2] - X[i1*N + (i2-1)] * A[i1*N + i2] / B[i1*N + (i2-1)];
			B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[i1*N + (i2-1)];
		}
	}
}


__global__ void adi_kernel2(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if ((i1 < _PB_N))
	{
		X[i1*N + (N-1)] = X[i1*N + (N-1)] / B[i1*N + (N-1)];
	}
}
	

__global__ void adi_kernel3(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i1 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i1 < _PB_N)
	{
		for (int i2 = 0; i2 < _PB_N-2; i2++)
		{
			X[i1*N + (N-i2-2)] = (X[i1*N + (N-2-i2)] - X[i1*N + (N-2-i2-1)] * A[i1*N + (N-i2-3)]) / B[i1*N + (N-3-i2)];
		}
	}
}


__global__ void adi_kernel4(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i2 < _PB_N)
	{
		X[i1*N + i2] = X[i1*N + i2] - X[(i1-1)*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
		B[i1*N + i2] = B[i1*N + i2] - A[i1*N + i2] * A[i1*N + i2] / B[(i1-1)*N + i2];
	}
}


__global__ void adi_kernel5(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i2 < _PB_N)
	{
		X[(N-1)*N + i2] = X[(N-1)*N + i2] / B[(N-1)*N + i2];
	}
}


__global__ void adi_kernel6(int n, DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* X, int i1)
{
	int i2 = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (i2 < _PB_N)
	{
		X[(N-2-i1)*N + i2] = (X[(N-2-i1)*N + i2] - X[(N-i1-3)*N + i2] * A[(N-3-i1)*N + i2]) / B[(N-2-i1)*N + i2];
	}
}





/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */





#include "../../common/polybench.c"
