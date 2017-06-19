#ifndef __RUNTIMESTATISTICSTESTER_CUH__
#define __RUNTIMESTATISTICSTESTER_CUH__

#include "UnitTesting/Tester.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "SATSolver/DataToDevice.cuh"

#define LARGE_TIME 200000000


class RuntimeStatisticsTester : public Tester
{
private:
    RuntimeStatistics *statistics;

    __device__ bool test_signal_start_stop();
    __device__ bool test_signal_job_start_stop();
    __device__ bool test_signal_decision_start_stop();
    __device__ bool test_signal_conflict_analysis_start_stop();
    __device__ bool test_signal_backtrack_start_stop();
    __device__ bool test_time(long long int before, long long int after, long long int control);

public:
    __device__ RuntimeStatisticsTester(DataToDevice data);
    __device__ void test_all();

};

#endif /* __RUNTIMESTATISTICSTESTER_CUH__ */
