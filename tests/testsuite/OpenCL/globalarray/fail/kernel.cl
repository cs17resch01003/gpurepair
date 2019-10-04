//xfail:REPAIR_ERROR
//--local_size=8 --num_groups=8 --no-inline

__constant int A[64] = { };

__kernel void globalarray(__global float* p) {
  int i = get_global_id(0) + 1;
  int a = A[i];

  char c;

  __constant char* cp = (__constant char*) A;

  c = cp[0];

  if(a == 0) {
    p[0] = get_global_id(0) + c;
  }
}
