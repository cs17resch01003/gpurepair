//pass
//--local_size=1024 --num_groups=64 --no-inline

#define sz get_local_size(0)
#define tid get_local_id(0)

__kernel void foo(__local int *A) {
  int temp;
  int i = 1;
  while(i < sz) {
    if(i < tid)
      temp = A[tid - i];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i < tid)
      A[tid] = A[tid] + temp;
    i = i * 2;
  }
}
