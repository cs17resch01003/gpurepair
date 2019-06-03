//pass
//--local_size=[16,16] --num_groups=[64,64,64] --no-inline




__kernel void foo(__global int* A) {

  // Only race free because of axioms provided by command-line options
  if(get_local_size(0) != 16 || get_local_size(1) != 16) {
    A[0] = get_local_id(0);
  }

}
