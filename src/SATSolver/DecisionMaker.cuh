#ifndef __DECISIONMAKER_CUH__
#define __DECISIONMAKER_CUH__

class DecisionMaker;

#include "VariablesStateHandler.cuh"
#include "DecisionStrategy/VSIDS.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "Utils/GPUVec.cuh"

class DecisionMaker
{
public:
    __device__ DecisionMaker(VariablesStateHandler *handler, const CUDAClauseVec *formula, size_t n_vars);
    __device__ Decision decide();
    __device__ void backtrack();
    __device__ void handle_learnt_clause(Clause c);
    __device__ void free_var(Var v);
    __device__ void block_var(Var v);

private:
    VariablesStateHandler *vars_handler;
    __device__ Lit new_literal();
    __device__ Decision new_decision();

#ifdef USE_VSIDS
    VSIDS vsids;
#endif

};

#endif /* __DECISIONMAKER_CUH__ */
