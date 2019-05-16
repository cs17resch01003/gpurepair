//xfail:NOT_ALL_VERIFIED
//--blockDim=2 --gridDim=1 --no-inline
//Write by thread .+kernel\.cu:8:21:

#include <cuda.h>

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] = curand_uniform(state);
}
