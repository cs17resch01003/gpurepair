//pass
//--blockDim=64 --gridDim=64 --no-inline

#include "cuda.h"

__device__ int bar () {
  return 0;
}

__global__ void foo() {
  bar ();
}

