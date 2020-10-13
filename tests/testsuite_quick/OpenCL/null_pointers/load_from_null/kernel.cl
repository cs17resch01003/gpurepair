//xfail:ASSERTION_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__global int *b)
{
  __global int *a = 0;
  int x = a[get_global_id(0)];
}

