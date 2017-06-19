#include "Heap.cuh"

__device__ Heap::Heap(int n_vars)
    : vector(n_vars)
    , vars_free_status { new bool[n_vars] }
{
    for (int i = 0; i < n_vars; i++) {
        vars_free_status[i] = true;
    }
}

__device__ bool Heap::is_free(Var var)
{
    //return vector.is_free(literal);
    return vars_free_status[var];
}

__device__ void Heap::set_free(Var var, bool free)
{
    //vector.set_free(literal, free);
    vars_free_status[var] = free;
}

__device__ int Heap::parent_index(int index)
{
    if (index == 0) {
        return -1;
    }
    int pindex;

    if (index % 2 == 0) {
        pindex = (index - 2) / 2;
    }
    else {
        pindex = (index - 1) / 2;
    }

    return pindex;
}

__device__ int Heap::left_child_index(int index)
{
    int lindex = 2 * index + 1;
    if (lindex >= vector.get_size()) {
        return -1;
    }
    else {
        return lindex;
    }
}
__device__ int Heap::right_child_index(int index)
{
    int lindex = 2 * index + +2;
    if (lindex >= vector.get_size()) {
        return -1;
    }
    else {
        return lindex;
    }
}

__device__ Lit Heap::get_max_lit()
{
    return vector.get_weighed_literal(0).literal;
}

__device__ void Heap::increment_lit(Lit literal, int increment)
{
#ifdef USE_ASSERTIONS
    assert(increment > 0);
#endif

    int index = vector.get_index(literal);
    int parent_i = parent_index(index);

    printf("index = %d, parent_index = %d\n", index, parent_i);

    vector.increment(index, increment);

    while (parent_i >= 0 &&
           vector.get_weighed_literal(index).weight >
           vector.get_weighed_literal(parent_i).weight) {
        vector.swap(index, parent_i);

        index = parent_i;
        parent_i = parent_index(index);
    }
}

__device__ void Heap::print()
{
    vector.print();
}
