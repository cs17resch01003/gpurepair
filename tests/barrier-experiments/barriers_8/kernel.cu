//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void race (int* __restrict__ A1, int* __restrict__ A2, int* __restrict__ A3, int* __restrict__ A4, int* __restrict__ A5, int* __restrict__ A6, int* __restrict__ A7, int* __restrict__ A8)
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

  temp = A5[idx + 1];
  A5[idx] = temp;

  temp = A6[idx + 1];
  A6[idx] = temp;

  temp = A7[idx + 1];
  A7[idx] = temp;

  temp = A8[idx + 1];
  A8[idx] = temp;
}