//xfail:ASSERTION_ERROR
//--local_size=2048 --num_groups=4 --no-inline

__kernel void foo(__global int* p) {

  __assert(__read_offset_bytes(p) == 42);

}
