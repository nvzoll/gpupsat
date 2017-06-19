#ifndef __HEAP_CUH__
#define __HEAP_CUH__

#include "SATSolver/SolverTypes.cuh"

struct weighed_literal {
    Lit literal;
    int weight;
    bool free;
    __device__ void print()
    {
        print_lit(literal);
        printf(" (%d),%s ", weight, free ? "f" : "l");
    }
};

class VarIndexedVector
{
private:
    unsigned int *indices;
    weighed_literal *vector;
    int n_vars;
    int size;

public:
    __device__ VarIndexedVector(int n_vars);
    __device__ void swap(unsigned int index_1, unsigned int index_2);
    __device__ void increment(int index, int increment);
    __device__ bool is_free(Lit literal);
    __device__ void set_free(Lit literal, bool free);

    __device__ int get_size() const { return size; }
    __device__ weighed_literal get_weighed_literal(Lit literal) const { return vector[literal.x]; }
    __device__ weighed_literal get_weighed_literal(int index) const { return vector[index]; }
    __device__ int get_index(Lit literal) const { return indices[literal.x]; }

    //Test method
    __device__ void print();
};

class Heap
{
private:
    VarIndexedVector vector;
    bool *vars_free_status;
    __device__ int parent_index(int index);
    __device__ int left_child_index(int index);
    __device__ int right_child_index(int index);
public:
    __device__ Heap(int n_vars);
    __device__ Lit get_max_lit();
    __device__ void increment_lit(Lit literal, int increment);
    __device__ bool is_free(Var var);
    __device__ void set_free(Var var, bool free);

    //Test method
    __device__ void print();
};

#endif /* __HEAP_CUH__ */
