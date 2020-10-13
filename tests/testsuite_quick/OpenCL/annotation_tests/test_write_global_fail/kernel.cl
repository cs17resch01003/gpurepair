//xfail:ASSERTION_ERROR
//--local_size=128 --num_groups=128 --no-inline

__kernel void foo(__global int* p, __global int* q) {

  p[get_global_id(0)] = q[get_global_id(0)];

  __assert(!__write(p));

}
