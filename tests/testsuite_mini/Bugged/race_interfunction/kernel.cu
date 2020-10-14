//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__device__ void write(int* A, int idx, int temp)
{
  A[idx] = temp;
}

__device__ int read(int* A, int idx)
{
  return A[idx + 1];
}

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  int temp = read(A, idx);
  write(A, idx, temp);
}