//xfail:REPAIR_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__local int* p) {

  p[get_local_id(0)] = get_global_id(0);

  barrier(0);

  p[get_local_id(0) + 1] = get_global_id(0);
}
