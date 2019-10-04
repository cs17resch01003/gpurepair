//xfail:ASSERTION_ERROR
//--local_size=1024 --num_groups=1 --no-inline

kernel void foo()
{
  atomic_inc((__global int*)0);
}
