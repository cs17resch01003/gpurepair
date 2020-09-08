//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void test_Prog(int *A, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;
  int alpha = A[idx + 1];

  if (idx >= 0)
  {
    int temp2 = A[idx + 2];
    A[idx] += temp2;
  }

  A[idx] += alpha;
}