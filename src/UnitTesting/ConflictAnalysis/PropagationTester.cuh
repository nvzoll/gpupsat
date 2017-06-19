#ifndef __PROPAGATIONTESTER_CUH__
#define __PROPAGATIONTESTER_CUH__

#include "SATSolver/VariablesStateHandler.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "ConflictAnalysis/ConflictAnalyzer.cuh"

class PropagationTester
{
private:
    VariablesStateHandler *vars_handler;
    const CUDAClauseVec *formula;

public:

    __device__ PropagationTester(const CUDAClauseVec *formula,
                                 VariablesStateHandler *vars_handler);

    __device__ bool test_single_propagation(ConflictAnalyzer *analyzer,
                                            sat_status& status);

    __device__ bool test_implication(Decision implication, ConflictAnalyzer *analyzer);

};
#endif /* __PROPAGATIONTESTER_CUH__ */
