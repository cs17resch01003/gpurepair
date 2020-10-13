//pass
//--local_size=64 --num_groups=64 --equality-abstraction --no-inline

__kernel void foo(__local short* p) {

  p[0]++;

}
