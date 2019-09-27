//xfail:NOT_ALL_VERIFIED
//--blockDim=32 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = threadIdx.x;

	int alpha=A[tid+1];
	if(tid>=0)
	{
		int temp2=A[tid+2];
		A[tid]+=temp2;
	}
	A[tid]+=alpha;
}
	