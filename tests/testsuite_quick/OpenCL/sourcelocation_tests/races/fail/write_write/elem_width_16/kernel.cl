//xfail:REPAIR_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__global int * p, __global short * q) {
 
  p = (__global int *)q;
  q[3] = get_local_id(0);
}


