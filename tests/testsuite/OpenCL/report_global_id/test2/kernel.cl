//xfail:REPAIR_ERROR
//--local_size=2 --num_groups=5

__kernel void foo(global int *p, int x) {
  if(get_global_id(0) == 0) {
    p[get_global_id(0)] = get_global_id(1);
  }
  if(get_global_id(0) == 7) {
    p[x] = 45;
  }
}
