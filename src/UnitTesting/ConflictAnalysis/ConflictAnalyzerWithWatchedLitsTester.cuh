#ifndef __CONFLICTANALYZERWITHWATCHEDLITSTESTER_CUH__
#define __CONFLICTANALYZERWITHWATCHEDLITSTESTER_CUH__

#include "UnitTesting/Tester.cuh"
#include "ConflictAnalysis/ConflictAnalyzerWithWatchedLits.cuh"
#include "SATSolver/DataToDevice.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "Utils/NodesRepository.cuh"
#include "Statistics/RuntimeStatistics.cuh"

class ConflictAnalyzerWithWatchedLitsTester : public Tester
{
private:
    DataToDevice *data;
    int input_file_number;

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> assumptions(0);
#else
    GPUStaticVec<Lit, 16> assumptions;
#endif

    __device__ ConflictAnalyzerWithWatchedLits generate_input(
        VariablesStateHandler& handler);

public:
    __device__ ConflictAnalyzerWithWatchedLitsTester(DataToDevice& data, int input_file_number);
    __device__ bool test_propagate_1_file1(ConflictAnalyzerWithWatchedLits *&input);
    __device__ bool test_propagate_2_file1(ConflictAnalyzerWithWatchedLits *&input);
    __device__ bool test_propagate_all_file2();
    __device__ bool test_implicate_and_backtrack_file1();
    __device__ bool test_reset_file1();
    __device__ void test_all();
};

#endif /* __CONFLICTANALYZERWITHWATCHEDLITSTESTER_CUH__ */
