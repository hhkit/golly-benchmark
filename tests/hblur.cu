__global__ void opt_blur(int *in, int *out) {
  __shared__ int cache[258];
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int lid = threadIdx.x + 1;
  cache[lid] = in[gid];
  if (lid == 1)
    cache[0] = in[gid];
  if (lid == 256)
    cache[lid + 1] = in[gid + 1];

  // __syncthreads(); // required, but forgotten by the programmer
  out[gid] = cache[lid - 1] + cache[lid] + cache[lid + 1];
}