//xfail:ASSERTION_ERROR
//--local_size=64 --num_groups=64 --no-inline

unsigned bar(unsigned, unsigned);

__kernel void foo(unsigned x, unsigned y) {
  unsigned z;
  z = bar(x, y);
  __assert(z >= x && z >= y);
}
