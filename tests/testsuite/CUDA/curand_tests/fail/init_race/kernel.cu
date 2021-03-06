//xfail:REPAIR_ERROR
//--blockDim=2 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void init_test(curandState *state, unsigned int *A) {
   curand_init(0, 0, 0, state);

   __syncthreads();

   if (threadIdx.x == 0) {
     A[0] = curand(state);
   }
}
