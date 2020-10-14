//pass
//--blockDim=512 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void curand_test(curandState *state, float *A) {
   A[threadIdx.x] = curand(&state[threadIdx.x]);
}
