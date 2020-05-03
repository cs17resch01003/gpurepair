//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = threadIdx.x;
	int alpha=0;
	if(tid%2==0)
	{
		alpha=A[tid+2];
	}
	__syncthreads();
	if(tid%6==0)
	{
		A[tid]=A[tid]+alpha;
	}
}