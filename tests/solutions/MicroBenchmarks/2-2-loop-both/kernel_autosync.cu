//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = threadIdx.x;
	for (int i = 0; i < N; ++i)
	{
		int tmp=A[tid+1];
__syncthreads();
		A[tid]=tmp;
__syncthreads();
	}	
}
