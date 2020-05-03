//xfail:TIMEOUT
//--local_size=[64,64] --global_size=[256,256]
                                                         
kernel void example(global float4 *G) {
    local float4 L[2][3][4][5];

    L[1][2][3][3].z = G[get_global_id(0)].x;

}
