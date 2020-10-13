//xfail:ASSERTION_ERROR
//--blockDim=1024 --gridDim=1 --no-inline

__constant__ int A[1024];

__global__ void foo(int *B) {
  A[threadIdx.x] = B[threadIdx.x];
}
