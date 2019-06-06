//xfail:REPAIR_ERROR
//--local_size=64 --num_groups=1 --no-benign-tolerance --no-inline

__kernel void foo(__local int* A, __local int* B, int i, int j) {
  A[0] = 0;
}
