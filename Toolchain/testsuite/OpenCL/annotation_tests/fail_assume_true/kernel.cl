//xfail:ASSERTION_ERROR
//--local_size=64 --num_groups=64 --no-inline

__kernel void foo()
{
  __assume(true);
  __assert(false);
}
