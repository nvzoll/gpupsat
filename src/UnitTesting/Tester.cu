#include "UnitTesting/Tester.cuh"

__host__ __device__ void Tester::process_test(bool return_value, char *method_name)
{
    if (return_value) {
        printf("\tMethod \"%s\" is SUCCESSFUL\n", method_name);
    }
    else {
        printf("\tMethod \"%s\" FAILED\n", method_name);
        n_errors++;
    }

    n_tests++;
}

__host__ __device__ Tester::Tester()
{
    n_errors = 0;
    n_tests = 0;
}

__host__ __device__ void Tester::print_summary()
{
    if (n_errors != 0) {
        printf("There were %d errors of %d\n", n_errors, n_tests);
    }
}
