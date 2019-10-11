//pass
//--blockDim=32 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void race (int* __restrict__ A1, int* __restrict__ A2, int* __restrict__ A3, int* __restrict__ A4)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;
  int temp = 0;

  temp = A1[idx + 1];
  A1[idx] = temp;

  temp = A2[idx + 1];
  A2[idx] = temp;

  temp = A3[idx + 1];
  A3[idx] = temp;

  temp = A4[idx + 1];
  A4[idx] = temp;
}