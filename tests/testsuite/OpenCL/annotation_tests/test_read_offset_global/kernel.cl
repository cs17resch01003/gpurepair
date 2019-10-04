//pass
//--local_size=2048 --num_groups=4 --no-inline




__kernel void foo(__global int* p, __global int* q) {

  if(get_global_id(0) == 0) {
    p[4] = q[5];
    __assert(__implies(__write(p) & __read(q),
          __read_offset_bytes(q) == __write_offset_bytes(p) + sizeof(int*)));
  }
}
