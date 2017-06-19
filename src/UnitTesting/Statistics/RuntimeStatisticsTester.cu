#include "RuntimeStatisticsTester.cuh"

__device__ RuntimeStatisticsTester::RuntimeStatisticsTester(DataToDevice data)
{
    statistics = data.get_statistics_ptr();
}

__device__ bool RuntimeStatisticsTester::test_time(int64_t before,
        int64_t after, int64_t control)
{
    if (before > after) {
        printf("\tBefore (%lld) should not be greater than after (%lld)\n",
               before, after);
        return false;
    }

    if (before < 0) {
        printf("\tBefore (%lld) should not be negative\n", before);
        return false;
    }

    if (after < 0) {
        printf("\tAfter (%lld) should not be negative\n", after);
        return false;
    }


    if (before > LARGE_TIME) {
        printf("\tBefore (%lld) seems to large to be true\n", before);
        return false;
    }

    if (after > LARGE_TIME) {
        printf("\tAfter (%lld) seems to large to be true\n", after);
        return false;
    }

    if (after - before > control ) {
        printf("After (%lld) minus before (%lld)"
               " should not be greater than control (%lld)\n",
               after, before, control);
        return false;

    }

    return true;
}

__device__ bool RuntimeStatisticsTester::test_signal_job_start_stop()
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int initial_jobs = (statistics->get_n_jobs_run())[index];

    int jobs_to_add = 43;

    int64_t before = statistics->get_total_time_solving();

    int64_t control = - clock64();

    for (int i = 0; i < jobs_to_add; i++) {
        statistics->signal_job_start();
        statistics->signal_job_stop();
    }

    control += clock64();

    int64_t after = statistics->get_total_time_solving();

    int resulting_jobs = (statistics->get_n_jobs_run())[index];

    if (resulting_jobs != initial_jobs + jobs_to_add) {
        printf("\tNumber of jobs added (%d) added to the number of initial jobs (%d)"
               " and the number of jobs (%d) do not equal\n", jobs_to_add, initial_jobs,
               resulting_jobs);
        return false;
    }

    if (!test_time(before, after, control)) {
        return false;
    }

    return true;


}
__device__ bool RuntimeStatisticsTester::test_signal_decision_start_stop()
{
    int decision_to_make = 30;

    int64_t before = statistics->get_total_deciding_time();

    int64_t control = - clock64();

    for (int i = 0; i < decision_to_make; i++) {
        statistics->signal_decision_start();
        statistics->signal_decision_stop();
    }

    control += clock64();

    int64_t after = statistics->get_total_deciding_time();

    return test_time(before, after, control);
}

__device__ bool RuntimeStatisticsTester::test_signal_conflict_analysis_start_stop()
{
    int decision_to_make = 30;

    int64_t before = statistics->get_total_conflict_analyzing_time();

    int64_t control = - clock64();

    int before_decision_level = 3;
    int after_decision_level = 1;
    int total_decision_level = 0;

    for (int i = 0; i < decision_to_make; i++) {
        statistics->signal_conflict_analysis_start(before_decision_level);
        statistics->signal_conflict_analysis_stop(after_decision_level);

        total_decision_level += before_decision_level - after_decision_level;
    }

    int backtracked_levels = statistics->get_total_backtracked_levels();

    if (total_decision_level != backtracked_levels) {
        printf("\tNumber of backtracked levels (%d) is not what it should be (%d).\n", backtracked_levels,
               total_decision_level);
        return false;
    }

    control += clock64();

    int64_t after = statistics->get_total_conflict_analyzing_time();

    if (!test_time(before, after, control)) {
        return false;
    }

    return true;
}

__device__ bool RuntimeStatisticsTester::test_signal_backtrack_start_stop()
{
    int decision_to_make = 30;

    int64_t before = statistics->get_total_backtracking_time();

    int64_t control = - clock64();

    int decisions_before = 3;
    int decisions_after = 1;
    int total_decisions_before = statistics->get_sum_of_decision_before_backtracking();
    int total_decisions_after = statistics->get_sum_of_decision_after_backtracking();

    for (int i = 0; i < decision_to_make; i++) {
        statistics->signal_backtrack_start(decisions_before);
        statistics->signal_backtrack_stop(decisions_after);

        total_decisions_before += decisions_before;
        total_decisions_after += decisions_after;

    }

    int sum_of_decisions_before = statistics->get_sum_of_decision_before_backtracking();
    int sum_of_decisions_after = statistics->get_sum_of_decision_after_backtracking();

    if (sum_of_decisions_before != total_decisions_before) {
        printf("\tThe number of decisions done before backtracking (%d)"
               " does not match the number made (%d)\n", sum_of_decisions_before, total_decisions_before);
        return false;
    }

    if (sum_of_decisions_after != total_decisions_after) {
        printf("\tThe number of decisions done after backtracking (%d)"
               " does not match the number made (%d)\n", sum_of_decisions_after, total_decisions_after);
        return false;
    }

    control += clock64();

    int64_t after = statistics->get_total_backtracking_time();

    if (!test_time(before, after, control)) {
        return false;
    }

    return true;
}

__device__ bool RuntimeStatisticsTester::test_signal_start_stop()
{

    int index = threadIdx.x + blockIdx.x * blockDim.x;

#define TESTS_SIZES 10

    uint64_t starts[TESTS_SIZES] = {1, 21, 54, 76, 89, 123, 456, 786, 1023, 1990};
    uint64_t stops[TESTS_SIZES] = {6, 32, 65, 81, 97, 234, 512, 989, 1235, 2314};
    //int64_t total_sum_calc[gridDim.x*blockDim.x];// = 0;
    uint64_t *total_sum_calc = new uint64_t[gridDim.x * blockDim.x]; // = 0;
    uint64_t total_sum_oracle = 0;
    //int occurrences_calc[gridDim.x*blockDim.x];// = 0;
    int *occurrences_calc = new int[gridDim.x * blockDim.x]; // = 0;
    int occurrences_oracle = 0;

    total_sum_calc[index] = 0;
    occurrences_calc[index] = 0;


    for (int i = 0; i < TESTS_SIZES; i++) {
        total_sum_oracle += stops[i] - starts[i];
        occurrences_oracle++;

        statistics->signal_start(total_sum_calc, occurrences_calc, starts[i]);
        statistics->signal_stop(total_sum_calc, stops[i]);

        if (total_sum_calc[index] != total_sum_oracle) {
            printf("\tSum calculated (%lld) and expected sum (%lld) do not match.\n",
                   total_sum_calc, total_sum_oracle);
            return false;
        }

        if (occurrences_calc[index] != occurrences_oracle) {
            printf("\tOccurrences calculated (%d) and expected occurrences (%d) do not match.\n",
                   occurrences_calc, occurrences_oracle);
            return false;
        }

    }

    delete [] total_sum_calc;
    delete [] occurrences_calc;

    return true;
}

__device__ void RuntimeStatisticsTester::test_all()
{
    printf("Runtime statistics tester:\n");
    Tester::process_test(test_signal_start_stop(), "Test Signal Start/Stop");
    Tester::process_test(test_signal_job_start_stop(), "Test Signal Job Start/Stop");
    Tester::process_test(test_signal_decision_start_stop(), "Test Signal Decision Start/Stop");
    Tester::process_test(test_signal_conflict_analysis_start_stop(), "Test Signal Conflict Start/Stop");
    Tester::process_test(test_signal_backtrack_start_stop(), "Test Signal Backtrack Start/Stop");
}
