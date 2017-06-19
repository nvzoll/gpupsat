#include "Heap.cuh"

__global__ void heap_test()
{

    Heap heap(10);
    heap.print();

    Lit l = mkLit(7, false);
    heap.increment_lit(l, 36);
    heap.print();

    l = mkLit(3, true);
    heap.increment_lit(l, 190);
    heap.print();

    l = mkLit(6, false);
    heap.increment_lit(l, 45);
    heap.print();

    printf("Done.\n");
}

int main_test_head()
{
    heap_test <<< 1, 1>>>();
    cudaDeviceReset();
}
