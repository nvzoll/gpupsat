#ifndef __CONFLICTANALYZERGENERICTESTER_CUH__
#define __CONFLICTANALYZERGENERICTESTER_CUH__

#include "ConflictAnalysis/ConflictAnalyzer.cuh"
#include "ConflictAnalysis/ConflictAnalyzerFullSearchSpace.cuh"
#include "ConflictAnalysis/ConflictAnalyzerWithWatchedLits.cuh"
#include "SATSolver/DataToDevice.cuh"
#include "UnitTesting/Tester.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "PropagationTester.cuh"

class ConflictAnalyzerGenericTester : public Tester
{
private:
    ConflictAnalyzer conflictAnalyzer;
    ConflictAnalyzerFullSearchSpace conflictAnalyzer_fullsearch;
    ConflictAnalyzerWithWatchedLits conflictAnalyzer_two_literals;
    VariablesStateHandler *vars_handler;
    const CUDAClauseVec *formula;
    int n_vars;
    int n_clauses;

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> assumptions(0);
#else
    GPUStaticVec<Lit, 16> assumptions;
#endif

public:
    __device__ ConflictAnalyzerGenericTester(DataToDevice& data,
            VariablesStateHandler *handler);

    __device__ bool test_init_states();
    __device__ bool test_resets();
    __device__ void test_all();
    __device__ bool test_propagates();
private:
    __device__ bool test_init_state(ConflictAnalyzer *analyzer);
    __device__ bool test_reset(ConflictAnalyzer *analyzer);
    __device__ bool test_propagate(ConflictAnalyzer *analyzer);
    __device__ bool perform_tests(
        bool (ConflictAnalyzerGenericTester::*funct_ptr)(ConflictAnalyzer *analyzer));
    __device__ bool change_analyzers_states();
};

#endif /* __CONFLICTANALYZERGENERICTESTER_CUH__ */
