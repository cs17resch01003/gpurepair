//pass
//--blockDim=32 --gridDim=2

#include <cuda.h>

__global__ void race (int* A, int* B)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  int temp = A[idx + 1];
  A[idx] = temp;
  
  int temp2 = 0;
  if (threadIdx.x != blockDim.x - 1)
  {
    temp2 = B[idx + 1];	
  }
  
  B[idx] = temp2;
}