//xfail:NOT_ALL_VERIFIED
//--blockDim=512 --gridDim=64 --loop-unwind=2 --no-inline
//kernel.cu: error: possible write-write race on B
#include <cuda.h>

extern "C" {

__global__ void helloCUDA(float *A)
{
    __shared__ float B[256];
    for(int i = 0; i < 10; i ++) {
        B[i] = A[i];
    }

}

}
