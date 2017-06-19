#include "Heap.cuh"

__device__ VarIndexedVector::VarIndexedVector(int n_vars)
    : n_vars { n_vars }
    , size { n_vars * 2 }
    , indices { new unsigned[size] }
    , vector { new weighed_literal[size] }
{
    for (int i = 0; i < n_vars; i++) {
        Lit lpos = mkLit(i, true);
        Lit lneg = mkLit(i, false);
        int pos_index = i * 2;
        int neg_index = i * 2 + 1;
        indices[lpos.x] = pos_index;
        indices[lneg.x] = neg_index;

        weighed_literal wlpos, wlneg;
        wlpos.literal = lpos;
        wlpos.weight = 0;
        wlneg.literal = lneg;
        wlneg.weight = 0;

        vector[pos_index] = wlpos;
        vector[neg_index] = wlneg;
    }
}

__device__ void VarIndexedVector::increment(int index, int increment)
{
    vector[index].weight += increment;
}

__device__ bool VarIndexedVector::is_free(Lit literal)
{
    weighed_literal wl = get_weighed_literal(literal);
    return wl.free;
}
__device__ void VarIndexedVector::set_free(Lit literal, bool free)
{
    int index = get_index(literal);
    vector[index].free = free;
}

__device__ void VarIndexedVector::swap(unsigned int index_1, unsigned int index_2)
{
    weighed_literal aux = vector[index_1];
    vector[index_1] = vector[index_2];
    vector[index_2] = aux;

    indices[vector[index_1].literal.x] = index_1;
    indices[vector[index_2].literal.x] = index_2;
}

__device__ void VarIndexedVector::print()
{
    printf("Indices[%d] = {", size);
    for (int i = 0; i < size; i++) {
        Lit l;
        l.x = i;

        print_lit(l);
        printf(" = (%d) ", indices[i]);
    }
    printf("}\nValues[%d] = {", size);
    for (int i = 0; i < size; i++) {
        printf("(%d) = ", i);
        vector[i].print();
    }
    printf("}\n");
}
