//xfail:REPAIR_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(int i, __global int *A)
{
  __global int *a;

  if (i == 0)
    a = A;
  else
    a = NULL;


  if (a != 0)
    A[get_global_id(0)] = get_global_id(0);
  else
    A[0] = get_global_id(0);
}

