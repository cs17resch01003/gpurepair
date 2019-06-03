//xfail:NOT_ALL_VERIFIED
//--local_size=64 --num_groups=64 --boogie-file=${KERNEL_DIR}/axioms.bpl --no-inline
//kernel.cl:10:3:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ with local id [\d+] in work group [\d]+[\s]+__assert\(z <= x\);

DECLARE_UF_BINARY(A, unsigned, unsigned, unsigned);

__kernel void foo(unsigned x, unsigned y) {
  unsigned z;
  z = A(x, y);
  __assert(z <= x);
}
