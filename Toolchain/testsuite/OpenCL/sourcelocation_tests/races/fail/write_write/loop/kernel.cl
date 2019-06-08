//xfail:REPAIR_ERROR
//--local_size=1024 --num_groups=1024 --no-inline

__kernel void foo(__local int* p) {

  for(int i = 0; i < 100; i++) {
		  if(get_local_id(0) == 5) {
			p[get_local_id(0)] = get_local_id(0);
		  }
		  if(get_local_id(0) == 4) {
			p[get_local_id(0)+1] = get_local_id(0);
		  }
  }


}
