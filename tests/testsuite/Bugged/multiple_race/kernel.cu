//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void race (int* A, int* B)
{
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  int idx = blockDim.x * bid + tid;

  int temp = A[idx + 1];
  A[idx] = temp;
  
  temp = B[idx + 1];
  B[idx] = temp;
}