#ifndef __LEARNTCLAUSESMANAGERTESTER_CUH__
#define __LEARNTCLAUSESMANAGERTESTER_CUH__

#include "ClauseLearning/LearntClausesManager.cuh"
#include "SATSolver/DataToDevice.cuh"
#include "UnitTesting/Tester.cuh"
#include "curand.h"

class LearntClausesManagerTester : public Tester
{
private:
    LearntClausesManager manager;
    WatchedClausesList watched_clauses;
    VariablesStateHandler handler;
    GPUStaticVec<Lit> assumptions;
    int n_vars;

    __device__ Clause generate_clause(int size);
    __device__ bool add_random_clause(int size);
    __device__ bool stress_test();
    __device__ bool check_clause_in_structure(Clause c);

public:
    __device__ LearntClausesManagerTester(DataToDevice& data);
    __device__ void test_all();
};

#endif /* __LEARNTCLAUSESMANAGERTESTER_CUH__ */
