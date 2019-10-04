//pass
//--blockDim=32 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void race (int* A)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  A[idx] = A[idx + 1];
}