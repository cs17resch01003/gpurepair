//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void test_Prog(int *A, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  for (int i = 0; i < N; ++i)
  {
    int tmp = A[idx + 1];
    A[idx] = tmp;
  }
}