//xfail:REPAIR_ERROR
//--local_size=2,3,4 --num_groups=5,6,7

__kernel void foo(global int *p, int x) {
  if(get_global_id(0) == 0 && get_global_id(1) == 1 && get_global_id(2) == 2) {
    p[get_global_id(0)] = get_global_id(1);
  }
  if(get_global_id(0) == 8 && get_global_id(1) == 13 && get_global_id(2) == 21) {
    p[x] = get_global_id(1);
  }
}
