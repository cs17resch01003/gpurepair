//xfail:REPAIR_ERROR
//--local_size=[64,64] --global_size=[256,256]
                                                         
kernel void example(global int *G) {
    local int L[2][3][4][5];

    L[1][2][3][3] = G[get_global_id(0)];

}
