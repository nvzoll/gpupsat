#ifndef __WATCHEDCLAUSESLISTTESTER_CUH__
#define __WATCHEDCLAUSESLISTTESTER_CUH__

#include "BCPStrategy/WatchedClausesList.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "Utils/GPULinkedList.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "UnitTesting/Tester.cuh"

class WatchedClausesListTester : public Tester
{
private:
    WatchedClausesList watched_clauses_list;
    VariablesStateHandler handler;
    const CUDAClauseVec *formula;
    GPUStaticVec<Lit> assumptions;

public:
    __device__ bool test_implication_from_clauses();


    __device__ WatchedClausesListTester(DataToDevice& data);
    __device__ void test_all();
};

#endif /* __WATCHEDCLAUSESLISTTESTER_CUH__ */
