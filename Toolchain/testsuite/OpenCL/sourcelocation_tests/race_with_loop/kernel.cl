//pass
//--local_size=1024 --num_groups=2 --no-inline

__kernel void foo(__local int* p) {

    int x = 0;
    for(int i = 0; i < 100; i++) {
        x += p[i];
        x += p[i+1];
    }

    p[get_local_id(0)] = x;

}
