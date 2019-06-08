//xfail:REPAIR_ERROR
//--local_size=64 --global_size=256

typedef struct {
    int x;
    int y;
} S;

kernel void example(global S *G) {

    G[3].y = get_global_id(0);

}
