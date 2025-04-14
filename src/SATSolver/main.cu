#include <assert.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "ErrorHandler/CudaMemoryErrorHandler.cuh"
#include "FileManager/CnfReader.h"
#include "FileManager/FileUtils.cuh"
#include "FileManager/ParametersManager.h"
#include "JobsManager/JobChooser.cuh"
#include "JobsManager/SimpleJobChooser.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "Utils/Stopwatch.cuh"

#include "Configs.cuh"
#include "Parallelizer.cuh"
#include "Results.cuh"

void print_info(int& n_blocks, int& n_threads, ParametersManager& pm, int& n_vars, int& n_clauses, FormulaData& fdata,
    int& max_implication_per_var)
{
    if (n_blocks < 1) {
        printf("Invalid number of blocks: %d\n", n_blocks);
    }
    if (n_threads < 1) {
        printf("Invalid number of threads: %d\n", n_threads);
        exit(1);
    }
    if (pm.get_verbosity_level() >= 1) {
        printf("Solver configuration:\n");
        printf("Input file: %s\n", pm.get_input_file());
        printf("Formula has %d vars and %d clauses\n", n_vars, n_clauses);
        printf("Variable '%d', the most frequent, has been found %d times.\n", fdata.get_most_common_var() + 1,
            fdata.get_frequency_of_most_common_var());
        if (pm.get_n_threads() == 1 && !pm.get_sequential_as_parallel()) {
            printf("Parallelization strategy: SEQUENTIAL RUN\n");
        } else {
            printf("Parallelization strategy: Divide and Conquer\n");
            printf("Number of blocks: %d\n", pm.get_n_blocks());
            printf("Number of threads: %d\n", pm.get_n_threads());

            switch (pm.get_choosing_strategy()) {
            case ChoosingStrategy::DISTRIBUTE_JOBS_PER_THREAD:
                printf("Job creation strategy: distribution per thread\n");
                break;
            case ChoosingStrategy::UNIFORM:
                printf("Job creation strategy: uniform\n");
                break;
            default:
                printf("Unknown job creation strategy!\n");
                break;
            }
        }

        printf("Conflict analysis: ON with");
        printf("out");
        printf(" forward edges\n");
        printf("Capacity of edges = %d\n", max_implication_per_var);
        printf("Assumptions are stored in a ");
        printf("statically");
        printf(" allocated vector.\n");
        printf("Formula clauses are stored in ");
        printf("several allocations.\n");
        printf("Unary clauses pre-processing is %s\n", pm.get_preprocess_unary_clauses() ? "ON" : "OFF");
        printf("Conflict analysis is ");
        switch (CONFLICT_ANALYSIS_STRATEGY) {
        case BASIC_SEARCH:
            printf("basic search\n");
            break;
        case FULL_SPACE_SEARCH:
            printf("full space search\n");
            break;
        case TWO_WATCHED_LITERALS:
            printf("two wached literals\n");
            break;
        default:
            printf("unknown!\n");
            break;
        }
    }
    printf("VSIDS is ");
#ifdef USE_VSIDS
    printf("ON\n");
#else
    printf("OFF\n");
#endif
    printf("Restart is ");
#ifdef USE_RESTART
    printf("ON\n");
#else
    printf("OFF\n");
#endif
    printf("Clause learning is ");
#ifdef USE_CLAUSE_LEARNING
    printf("ON with learnt clause capacity of %d\n", MAX_LEARNT_CLAUSES_PER_THREAD);
#else
    printf("OFF\n");
#endif
    printf("Simple jobs generation is ");
#ifdef USE_SIMPLE_JOBS_GENERATION
    printf("ON\n");
#else
    printf("OFF\n");
#endif
}

int main(int argc, char *argv[])
{
    check(cudaSetDeviceFlags(cudaDeviceMapHost), "Setting device flag");

    ParametersManager pm(argc, argv);

    if (!file_exists(pm.get_input_file())) {
        printf("The specified CNF file (%s) was not found!\n", pm.get_input_file());
        exit(-1);
    }

    int lines = n_lines(pm.get_input_file());

    int n_vars;
    int n_clauses;

    FormulaData fdata(lines, pm.get_preprocess_unary_clauses());

    bool success = CnfManager().read_cnf(pm.get_input_file(), fdata);
    if (!success) {
        printf("Error parsing inputs.\n");
        exit(-1);
    }

    if (fdata.get_n_vars() < MIN_VARIABLES_TO_PARALLELIZE
        && (pm.get_n_blocks() > 1 || pm.get_n_threads() > 1 || pm.get_sequential_as_parallel())) {
        printf("Warning: There are %d vars in the formula and at least %d "
               "are necessary to parallelize. Forcing sequential execution!\n",
            fdata.get_n_vars(), MIN_VARIABLES_TO_PARALLELIZE);
        pm.force_sequential_configuration();
    }

    const CUDAClauseVec formula = fdata.get_formula_dev();
    const std::vector<Clause> formula_host = *(fdata.get_formula_host());
    n_vars = fdata.get_n_vars();
    n_clauses = fdata.get_n_clauses();

    int n_threads = pm.get_n_threads();
    int n_blocks = pm.get_n_blocks();

    // This holds the max implication a var may have, to set the capacity of edges.
    int max_implication_per_var = std::max(fdata.get_largest_clause_size(), MIN_IMPLICATION_PER_VAR);

    print_info(n_blocks, n_threads, pm, n_vars, n_clauses, fdata, max_implication_per_var);

    if (fdata.get_status_after_preprocessing() != sat_status::UNDEF) {
        if (pm.get_verbosity_level() >= 1) {
            printf("Solved in pre-processing.\n");
        }

        Results res(n_vars, false);

        res.set_host_status(fdata.get_status_after_preprocessing());
        res.print_results(fdata.get_solved_literals(), formula_host);

        return 0;
    }

    std::vector<Lit> const& solved_lits = fdata.get_solved_literals();
    GPUVec<Var> dead_vars_dev(solved_lits.size());
    std::vector<Var> dead_vars_host;

    for (Lit lit : solved_lits) {
        Var v = var(lit);

        dead_vars_dev.add(v);
        dead_vars_host.push_back(v);
    }

    // Original assumptions!
    GPUVec<Lit> assumptions(0);

    Results *results;

    int *state_host_ptr;
    int *state_dev_ptr;
    DataToDevice *data_host_ptr;
    DataToDevice *data_dev_ptr;

    check(cudaHostAlloc(&state_host_ptr, sizeof(int), cudaHostAllocMapped), "Alloc pinned memory");
    check(cudaHostGetDevicePointer(&state_dev_ptr, state_host_ptr, 0), "Retrieve state dev_ptr");

    check(cudaHostAlloc(&data_host_ptr, (sizeof(DataToDevice) * n_blocks * n_threads), cudaHostAllocMapped),
        "Alloc pinned memory");
    check(cudaHostGetDevicePointer(&data_dev_ptr, data_host_ptr, 0), "Retrieve state dev_ptr");
    *state_host_ptr = INT_MAX;

    check(cudaDeviceSetLimit(cudaLimitStackSize, DEVICE_THREAD_STACK_LIMIT), "Set stack limit");

    check(cudaThreadSetLimit(cudaLimitMallocHeapSize, DEVICE_THREAD_HEAP_LIMIT), "Set heap limit");

    float elapsedTime = 0.;
    CudaStopwatch stopwatch;

    if (n_threads == 1 && n_blocks == 1 && !pm.get_sequential_as_parallel()) {
        if (pm.get_verbosity_level() >= 1) {
            printf("Number of jobs = 1\n");
        }
#ifdef DEBUG
        printf("About to call sequential kernel!\n");
#endif
        DataToDevice::numbers n = { n_vars, n_clauses, 0, 1, 1, max_implication_per_var };

        DataToDevice data(
            formula, dead_vars_dev, RuntimeStatistics(n.blocks, n.threads, nullptr), n, DataToDevice::atomics());

        data.prepare_sequencial();

        stopwatch.start();
        run_sequential<<<1, 1>>>(data, state_dev_ptr);
        elapsedTime = stopwatch.stop();

        results = data.get_results_ptr();
    } else {
#ifdef USE_SIMPLE_JOBS_GENERATION
        SimpleJobChooser chooser(n_vars, dead_vars_host);
#else
        MaxClauseJobChooser chooser(
            formula_host, n_vars, dead_vars_dev.size_of(), n_threads, n_blocks, pm.get_choosing_strategy());
#endif
        chooser.evaluate();

        int n_jobs = chooser.get_n_jobs();

        if (pm.get_verbosity_level() >= 1) {
            printf("Number of jobs = %d\n", n_jobs);
        }

        DataToDevice::atomics atomics = { 0 };
        unsigned zero = 0;

        check(cudaMalloc(&atomics.next_job, sizeof(unsigned)), "Allocating counter");
        check(cudaMemcpy(atomics.next_job, &zero, sizeof(unsigned), cudaMemcpyHostToDevice), "Zeroing counter");

        check(cudaMalloc(&atomics.completed_jobs, sizeof(unsigned)), "Allocating counter");
        check(cudaMemcpy(atomics.completed_jobs, &zero, sizeof(unsigned), cudaMemcpyHostToDevice), "Zeroing counter");

        DataToDevice::numbers n = { n_vars, n_clauses, n_jobs, n_blocks, n_threads, max_implication_per_var };

        RuntimeStatistics stat(n.blocks, n.threads, atomics.completed_jobs);

        DataToDevice data(formula, dead_vars_dev, stat, n, atomics);
        data.prepare_parallel(*dynamic_cast<JobChooser *>(&chooser)
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
                                  ,
            assumptions
#endif
        );
        memcpy(data_host_ptr, &data, sizeof(data));

        KernelContextStorage thread_contexts;

        check(cudaMallocPitch(&thread_contexts.data, &thread_contexts.pitch, n_threads * sizeof(void *), n_blocks),
            "Allocate thread contexts");

        printf("About to invoke kernel...\n");

        parallel_kernel_init<<<n_blocks, n_threads>>>(data_dev_ptr, thread_contexts);

        size_t call = 1;
        while (true) {
            stopwatch.start();
            parallel_kernel<<<n_blocks, n_threads>>>(thread_contexts, state_dev_ptr);
            elapsedTime += stopwatch.stop();

            call++;

            if (*state_host_ptr != INT_MAX || call == SIZE_MAX) {
                break;
            }
        }

        printf("Kernel was invoked %zu times\n", (call - 1));

        if (call == SIZE_MAX) {
            printf("Max invocations reached\n");
            cudaDeviceReset();
            return -1;
        }

        DataToDevice *winner = data_host_ptr;

        parallel_kernel_retrieve_results<<<n_blocks, n_threads>>>(*winner, thread_contexts);

        results = winner->get_results_ptr();

#ifdef ENABLE_STATISTICS
        int solved_jobs = winner->get_statistics().get_all_threads_total_completed_jobs();

        if (results->get_status() == sat_status::UNSAT) {
            assert(solved_jobs == n_jobs);
        }

        printf("Jobs size = %d\n", n_jobs);
        printf("There were %d jobs created.\nThere were %d solved jobs\n", n_jobs, solved_jobs);

        cudaDeviceSynchronize();

        // winner->get_statistics().print_function_time_statistics();
#endif // ENABLE_STATISTICS
    }

    printf("Total time on GPU: %f ms\n", elapsedTime);

    results->print_results(fdata.get_solved_literals(), formula_host);

    if (pm.get_write_log()) {
        char buf[256];
        std::ofstream out("autolog.txt", std::ios_base::app);
        std::snprintf(buf, sizeof(buf), "%s,%d,%d,%f\n", pm.get_input_file(), n_threads, n_blocks, elapsedTime);
        out << buf;
    }

    cudaDeviceReset();

    return 0;
}
