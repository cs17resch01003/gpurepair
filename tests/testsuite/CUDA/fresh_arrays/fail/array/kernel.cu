//xfail:REPAIR_ERROR
//--blockDim=2048 --gridDim=64

struct s {
  float *p[42];
};

__global__ void foo(s q) {
  __requires_fresh_array(q.p[4]);
  q.p[4][0] = threadIdx.x;
}
