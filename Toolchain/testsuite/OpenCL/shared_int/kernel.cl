//xfail:REPAIR_ERROR
//--local_size=64 --num_groups=64 --no-inline

__kernel void foo() {
  __local int a;

  a = get_local_id(0);

}

