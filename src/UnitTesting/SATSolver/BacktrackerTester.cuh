#ifndef __BACKTRACKERTESTER_CUH__
#define __BACKTRACKERTESTER_CUH__

#include "SATSolver/VariablesStateHandler.cuh"
#include "SATSolver/Backtracker.cuh"
#include "SATSolver/Configs.cuh"
#include "Utils/GPUVec.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "SATSolver/DataToDevice.cuh"
#include "UnitTesting/Tester.cuh"

class BacktrackerTester : public Tester
{

private:
    VariablesStateHandler handler;
    Backtracker backtracker;

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> assumptions;
#else
    GPUStaticVec<Lit> assumptions;
#endif
    int n_vars;

    __device__ void reset();
    __device__ void add_decisions(bool randomly_branched);

    __device__ bool test_non_chronological_backtracking();
    __device__ bool test_chronological_backtracking();


public:
    __device__ BacktrackerTester(DataToDevice& data);
    __device__ void test_all();


};


#endif /* __BACKTRACKERTESTER_CUH__ */
