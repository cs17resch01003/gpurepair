//xfail:REPAIR_ERROR
//--local_size=64 --num_groups=64 --no-inline

__kernel void foo(__global int3 *n)
{
  n[200] = get_global_id(0);
}
