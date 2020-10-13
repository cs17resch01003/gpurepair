//xfail:ASSERTION_ERROR
//--local_size=64 --num_groups=64 --clang-opt=-Wno-uninitialized --no-inline

__kernel void foo() {
  int x;
  __assert(x == 0);
}


