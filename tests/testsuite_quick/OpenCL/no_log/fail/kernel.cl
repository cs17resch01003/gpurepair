//xfail:ASSERTION_ERROR
//--local_size=16 --num_groups=1 --only-log --no-inline

__kernel void foo(__local int* A) {
  A[get_local_id(0)] = get_local_id(0);
  __assert(!__write(A));
}
