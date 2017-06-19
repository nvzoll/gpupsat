#ifndef __TESTER_CUH__
#define __TESTER_CUH__

#include "SATSolver/DataToDevice.cuh"
#include "TestConfigs.cuh"
#include <stdlib.h>

class Tester
{
protected:
    char tester_name[MAX_TESTER_NAME];

    int n_errors;
    int n_tests;

public:
    __device__ Tester();
    __device__ ~Tester() { };

    __device__ virtual void test_all() = 0;
    __device__ void process_test(bool return_value, char *method_name);

    __device__ void print_summary();

    __host__ __device__
    int get_n_errors() const { return n_errors; }

    __host__ __device__
    int get_n_tests() const { return n_tests; }
};

#endif /* __TESTER_CUH__ */
