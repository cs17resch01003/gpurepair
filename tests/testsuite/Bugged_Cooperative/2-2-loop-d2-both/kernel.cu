//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void test_Prog(int *A, int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  for(int d = N/2; d > 0; d = d / 2)
  {
    int tmp = A[idx + d];
    for (int i = 0; i < N; ++i)
    {
      int tmp2 = A[idx];
      int t2 = tmp2;
      int t32 = t2;
	  
      if (idx < d) {
        A[idx] = tmp + t32;
      }
    }
  }
}