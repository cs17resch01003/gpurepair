//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void test_Prog(int *A, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;
  for (int d = N/2; d > 0; d = d / 2) {
    if (idx < d) {
      A[idx] += A[idx + d];
    }
  }
}