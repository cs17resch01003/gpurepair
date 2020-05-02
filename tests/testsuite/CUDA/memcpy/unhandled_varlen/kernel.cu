//xfail:BUGLE_ERROR
//--gridDim=1 --blockDim=2 --no-inline

//This kernel is racy. However, variable-length memcpys are not supported.
//Expect error at Bugle stage.

#define memcpy(dst, src, len) __builtin_memcpy(dst, src, len)

typedef struct {
  short x;
  short y;
  char z;
} s_t; //< sizeof(s_t) == 6

__global__ void overstep(s_t *in, s_t *out, size_t len) {
  memcpy(&out[threadIdx.x], &in[threadIdx.x], len);
}
