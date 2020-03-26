//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void test_Prog(int *A,int *B, int N) { 
	const int tid = threadIdx.x;
	int tmp=A[tid+1];
	int tmp2=B[tid+1];
__syncthreads();
	B[tid]=tmp2+tmp;
	A[tid]=tmp2-tmp;	
}