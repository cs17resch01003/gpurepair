//xfail:NOT_ALL_VERIFIED
//--blockDim=1024 --gridDim=1 --warp-sync=16 --no-inline

#include <cuda.h>

__global__ void shuffle (int* A)
{
	int tid = threadIdx.x;
	int warp = tid / 32;
	int* B = A + (warp*32);
	A[tid] = B[(tid + 1)%32];
}
