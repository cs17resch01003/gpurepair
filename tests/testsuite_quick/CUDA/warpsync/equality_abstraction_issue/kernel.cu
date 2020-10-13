//xfail:ASSERTION_ERROR
//--warp-sync=32 --blockDim=32 --gridDim=1 --equality-abstraction --no-inline

__global__ void foo(int * A) {
    A[0] = 1;
    A[1] = 1;
    A[2] = 1;
    A[threadIdx.x] = 0;
    __assert(A[0] == 1 | A[1] == 1 | A[2] == 1);
}
