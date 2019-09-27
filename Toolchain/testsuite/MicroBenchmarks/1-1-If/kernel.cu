//xfail:NOT_ALL_VERIFIED
//--blockDim=32 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = threadIdx.x;
	int alpha=0;
	if(tid%2==0)
	{
		alpha=A[tid+2];
	}
	if(tid%6==0)
	{
		A[tid]=A[tid]+alpha;
	}
}