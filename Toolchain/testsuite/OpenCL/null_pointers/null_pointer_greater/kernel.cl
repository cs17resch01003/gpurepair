//xfail:NOT_ALL_VERIFIED
//--local_size=1024 --num_groups=1024 --no-inline
//error: possible write-write race

__kernel void foo(int i, __global int *A, __global bool *B)
{
  __global int *a;

  if (i == 0)
    a = A;
  else
    a = NULL;


  if (a > (__global int *)0)
    A[get_global_id(0)] = get_global_id(0);
  else
    A[0] = get_global_id(0);
}

