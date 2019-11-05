//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void race (int* __restrict__ A1, int* __restrict__ A2, int* __restrict__ A3, int* __restrict__ A4, int* __restrict__ A5, int* __restrict__ A6, int* __restrict__ A7, int* __restrict__ A8, int* __restrict__ A9, int* __restrict__ A10, int* __restrict__ A11, int* __restrict__ A12, int* __restrict__ A13, int* __restrict__ A14, int* __restrict__ A15, int* __restrict__ A16, int* __restrict__ A17, int* __restrict__ A18, int* __restrict__ A19, int* __restrict__ A20, int* __restrict__ A21, int* __restrict__ A22, int* __restrict__ A23, int* __restrict__ A24, int* __restrict__ A25, int* __restrict__ A26, int* __restrict__ A27, int* __restrict__ A28, int* __restrict__ A29, int* __restrict__ A30, int* __restrict__ A31, int* __restrict__ A32)
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

  temp = A9[idx + 1];
  A9[idx] = temp;

  temp = A10[idx + 1];
  A10[idx] = temp;

  temp = A11[idx + 1];
  A11[idx] = temp;

  temp = A12[idx + 1];
  A12[idx] = temp;

  temp = A13[idx + 1];
  A13[idx] = temp;

  temp = A14[idx + 1];
  A14[idx] = temp;

  temp = A15[idx + 1];
  A15[idx] = temp;

  temp = A16[idx + 1];
  A16[idx] = temp;

  temp = A17[idx + 1];
  A17[idx] = temp;

  temp = A18[idx + 1];
  A18[idx] = temp;

  temp = A19[idx + 1];
  A19[idx] = temp;

  temp = A20[idx + 1];
  A20[idx] = temp;

  temp = A21[idx + 1];
  A21[idx] = temp;

  temp = A22[idx + 1];
  A22[idx] = temp;

  temp = A23[idx + 1];
  A23[idx] = temp;

  temp = A24[idx + 1];
  A24[idx] = temp;

  temp = A25[idx + 1];
  A25[idx] = temp;

  temp = A26[idx + 1];
  A26[idx] = temp;

  temp = A27[idx + 1];
  A27[idx] = temp;

  temp = A28[idx + 1];
  A28[idx] = temp;

  temp = A29[idx + 1];
  A29[idx] = temp;

  temp = A30[idx + 1];
  A30[idx] = temp;

  temp = A31[idx + 1];
  A31[idx] = temp;

  temp = A32[idx + 1];
  A32[idx] = temp;
}