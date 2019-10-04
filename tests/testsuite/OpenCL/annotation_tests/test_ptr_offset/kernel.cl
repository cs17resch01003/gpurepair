//pass
//--local_size=64 --num_groups=64 --equality-abstraction --no-inline


void bar(__global int* p) {
  __requires(!__read(p));
  __requires(!__write(p));
  __requires(__ptr_offset_bytes(p) == 4);

  __global int* q;

  q = p + 1;

  q[0] = 0;

  __assert(__ptr_offset_bytes(q) == 8);
}

__kernel void foo(__global int* a) {
  bar(a + 1);
}
