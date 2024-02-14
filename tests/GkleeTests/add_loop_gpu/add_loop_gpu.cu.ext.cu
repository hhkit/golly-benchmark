/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */
#include <cstdio>

//#include "../common/book.h"

#define N   10

__global__ void add( int *a, int *b, int *c ) {
  int tid = blockIdx.x;    // this thread handles the data at its thread id
  if (tid < N)
    c[tid] = a[tid] + b[tid];
}


