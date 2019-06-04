//pass
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__local int* a) {


  if (get_local_id(0) == 3) {
    barrier(CLK_GLOBAL_MEM_FENCE);
  }

  a[get_local_id(0)] = get_local_id(0);

}
