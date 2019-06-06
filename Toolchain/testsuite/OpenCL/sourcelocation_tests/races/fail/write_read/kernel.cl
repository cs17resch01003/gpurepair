//pass
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__local int* a, __local int* b) {

  a[get_local_id(0)] = get_local_id(0);

  b[get_local_id(0)] = a[3];

}
