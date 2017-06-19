#ifndef __GPULINKEDLISTTESTER_CUH__
#define __GPULINKEDLISTTESTER_CUH__

#include "UnitTesting/Tester.cuh"
#include "Utils/GPULinkedList.cuh"

#define NUMBER_OF_TESTS 1000000

class GPULinkedListTester : public Tester
{
private:
    __device__ bool stress_test();
public:
    __device__ GPULinkedListTester();
    __device__ void test_all();
};

#endif /* __GPULINKEDLISTTESTER_CUH__ */
