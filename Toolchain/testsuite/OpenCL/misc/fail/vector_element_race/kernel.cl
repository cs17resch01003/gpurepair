//xfail:REPAIR_ERROR
//--local_size=64 --global_size=256
                                                         
kernel void example(global float4 *G) {

    G[3].y = (float)get_global_id(0);

}
