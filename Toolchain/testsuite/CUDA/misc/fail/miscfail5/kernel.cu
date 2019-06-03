//xfail:ASSERTION_ERROR
//--gridDim=1 --blockDim=4 --no-inline

__constant__ int global_constant[4];

__global__ void constant(int *in) {
    global_constant[threadIdx.x] = in[threadIdx.x];
}
