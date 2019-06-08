//xfail:REPAIR_ERROR
//--local_size=16 --num_groups=1 --loop-unwind=10 --no-inline

__kernel void foo(__local int* A) {

    if(get_local_id(0) != 0) {
      A[get_local_id(0)] = get_local_id(0);
    }

    for(int i = 0; i < 100; i++) {

        if(i == 1) {
            A[0] = get_local_id(0);
        }

    }


}
