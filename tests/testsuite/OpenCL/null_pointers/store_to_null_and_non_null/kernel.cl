//xfail:ASSERTION_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__global int *A)
{
  __global int *a = NULL;

  a[get_global_id(0)] = get_global_id(0);
  A[get_global_id(0)] = get_global_id(0);
}

