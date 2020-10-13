//xfail:ASSERTION_ERROR
//--local_size=20 --num_groups=16 --check-array-bounds

__kernel void foo() {
  local int L[64];
  ((local char*)L)[get_global_id(0)] = get_global_size(0);
}
