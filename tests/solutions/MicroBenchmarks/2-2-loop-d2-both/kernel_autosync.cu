//pass
//--blockDim=32 --gridDim=1

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	for(int d = N/2; d > 0; d = d / 2) 
	{
__syncthreads();
		int tmp=A[tid + d];
		
		for (int i = 0; i < N; ++i)
		{
			int tmp2=A[tid];
			int t2=tmp2;
			int t32=t2;
__syncthreads();
			if (tid < d) {
			    A[tid] = tmp + t32;
			  }
		}	
	}

}