//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void test_Prog(int *A, int *B, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  int tmp = A[idx + 1];
  int tmp2 = B[idx + 1];
  B[idx] = tmp2 + tmp;
  A[idx] = tmp2 - tmp;
}