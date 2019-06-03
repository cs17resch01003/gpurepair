//xfail:REPAIR_ERROR
//--blockDim=256 --gridDim=2 --no-inline

#include <cuda.h>

__global__ void curand_test(curandStateMtgp32_t *state, float *A) {
  if (threadIdx.x == 0) {
    A[blockIdx.x] = curand(state);
  }
}
