//xfail:REPAIR_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__global int* p) {
  p[get_local_id(0)] = get_group_id(0);
}
