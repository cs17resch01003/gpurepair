//xfail:NOT_ALL_VERIFIED
//--local_size=128 --num_groups=128 --no-inline
//kernel.cl:13:[\d]+:[\s]+error:[\s]+this assertion might not hold for work item [\d]+ with local id [\d]+ in work group [\d]+[\s]+__assert\(!__write\(p\)\);





__kernel void foo(__local int* p, __local int* q) {

  p[get_local_id(0)] = q[get_local_id(0)];

  __assert(!__write(p));

}
