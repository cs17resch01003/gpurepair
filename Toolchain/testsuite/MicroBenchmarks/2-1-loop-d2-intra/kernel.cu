//pass
//--blockDim=32 --gridDim=1 --no-inline

#include <cuda.h>

__global__ void test_Prog(int *A, int N) { 
	const int tid = threadIdx.x;
	for(int d = N/2; d > 0; d = d / 2){
		for(int i=0;i<100;i++) {
		  if (tid < d && i==0) {
		    A[tid] = A[tid + d];
		  }
		}
	}
}
