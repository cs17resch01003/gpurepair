//xfail:REPAIR_ERROR
//--local_size=65 --num_groups=1 --no-inline

__kernel void foo(__local int* A, int i) {
  A[i] = get_local_id(0) / 64;
}
