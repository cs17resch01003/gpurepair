//xfail:REPAIR_ERROR
//--blockDim=1024 --gridDim=1024

__device__  double C[1024][0][1024];

__global__ void foo(double *H) {
  C[threadIdx.x][threadIdx.y][threadIdx.z] = H[threadIdx.x];
}
