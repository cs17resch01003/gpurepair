//pass
//--blockDim=10 --gridDim=64 --no-inline

#include "cuda.h"


__global__ void foo() {

  __shared__ int A[10];

  A[threadIdx.x] = 0;

}
