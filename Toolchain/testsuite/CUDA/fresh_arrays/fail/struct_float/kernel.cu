//xfail:REPAIR_ERROR
//--blockDim=2048 --gridDim=64

struct s {
  float *p;
};

__global__ void foo(s q) {
  __requires_fresh_array(q.p);
  q.p[0] = threadIdx.x;
}
