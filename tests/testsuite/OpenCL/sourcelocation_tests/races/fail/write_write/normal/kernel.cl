//xfail:REPAIR_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__local int* a) {

  a[get_local_id(0)] = get_local_id(0);
  if(get_local_id(0) == 0)
    a[7] = 0;

}


