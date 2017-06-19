#ifndef __CUDALISTGRAPHTESTER_CUH__
#define __CUDALISTGRAPHTESTER_CUH__

#include "UnitTesting/TestConfigs.cuh"

#include "ConflictAnalysis/CUDAListGraph.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "SATSolver/DataToDevice.cuh"
#include "UnitTesting/Tester.cuh"

class CUDAListGraphTester : public Tester
{
private:
    CUDAListGraph graph;
    int n_vars;
    int n_clauses;

public:
    __device__ CUDAListGraphTester(DataToDevice& data);
    __device__ bool test_initial_state();
    __device__ bool test_set_and_is_set();
    __device__ bool test_backtrack_to();
    __device__ bool test_link_and_neighbors_methods();
    __device__ bool test_flag_unflag();
    __device__ bool test_link_linked();
    __device__ bool stress_test();
    __device__ void test_all();

};

#endif /* __CUDALISTGRAPHTESTER_CUH__ */
