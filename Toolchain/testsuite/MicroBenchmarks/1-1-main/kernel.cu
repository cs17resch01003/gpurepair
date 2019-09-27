//xfail:NOT_ALL_VERIFIED
//--blockDim=32 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = threadIdx.x;
	int tmp=A[tid+1];
	tmp=tmp+11;
	A[tid]+=tmp;	
}