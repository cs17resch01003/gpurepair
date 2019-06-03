//xfail:REPAIR_ERROR
//--blockDim=2048 --gridDim=64

struct s {
  float *p;
};

struct t {
  s v;
};

__global__ void foo(t q) {
  __requires_fresh_array(q.v.p);
  q.v.p[0] = threadIdx.x;
}
