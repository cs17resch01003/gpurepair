//xfail:REPAIR_ERROR
//--blockDim=128 --gridDim=128 --warp-sync=32 --no-inline

__global__ void foo(int* A) {

    A[ blockIdx.x*blockDim.x + threadIdx.x ] += (A[ (blockIdx.x + 1)*blockDim.x + threadIdx.x ]);

}
