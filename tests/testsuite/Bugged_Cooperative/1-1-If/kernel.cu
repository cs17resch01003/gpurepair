//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void test_Prog(int *A, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;
  int alpha = 0;
  
  if (idx % 2 == 0)
  {
    alpha = A[tid + 2];
  }
  
  if (idx % 6 == 0)
  {
    A[tid] = A[tid] + alpha;
  }
}