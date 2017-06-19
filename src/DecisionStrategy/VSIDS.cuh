#ifndef __VSIDS_CUH__
#define __VSIDS_CUH__

#include <stdint.h>
#include <curand_kernel.h>

#include "SATSolver/SolverTypes.cuh"

class VSIDS
{
private:
    struct vsids_var {
        int positive_lit_sum;
        int negative_lit_sum;
        bool free;
    };

    const int decay_factor;
    const int clauses_before_decaying;
    const float random_decision_frequency;

    size_t n_vars;

    size_t n_decisions = 0;
    size_t n_learnt_clauses = 0;

    vsids_var *vars;

    curandState randState;

    __device__ void initCurand(uint32_t seed);

    __device__ void decay();
    __device__ void increment(Lit literal);
    __device__ Var next_random_var();
    __device__ bool next_random_polarity();
    __device__ Lit next_random_literal();
    __device__ Lit next_higher_literal();

public:
    __device__ VSIDS(size_t n_vars);

    __device__ void free_var(Var v);
    __device__ void block_var(Var v);
    __device__ void handle_clause(Clause c);
    __device__ Lit next_literal();

    // Test methods
    __device__ void print();
    __device__ bool is_free(Var v);
};

#endif /* __VSIDS_CUH__ */
