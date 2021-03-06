//xfail:ASSERTION_ERROR
//--local_size=256 --num_groups=2 --vcgen-op=/checkArrays:A --infer-info

__kernel void test(__global double *A, __global double *B) {
  A[get_global_id(0)] = 0;

  for (int i = 0;
       __global_invariant(__implies(__same_group & __write(B), false)),
       i < 42; ++i) {
    B[i] = get_global_id(0);
  }

  __assert(false);
}
