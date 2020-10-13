//xfail:REPAIR_ERROR
//--blockDim=2 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] = curand_uniform(state);
}
