#include <assert.h>
#include <cmath> // Include for std::pow if needed by choosers
#include <fstream> // Include for std::ofstream
#include <new> // Include for placement new
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <cuda/std/cassert>
#include <thrust/device_malloc.h>
#include <thrust/device_new.h>
#include <thrust/device_vector.h>
#include <vector>

#include "ErrorHandler/CudaMemoryErrorHandler.cuh"
#include "FileManager/CnfReader.h"
#include "FileManager/FileUtils.cuh"
#include "FileManager/ParametersManager.h"
#include "JobsManager/JobChooser.cuh"
#include "JobsManager/SimpleJobChooser.cuh"
// #include "JobsManager/MaxClauseJobChooser.cuh" // Include if needed
#include "Statistics/RuntimeStatistics.cuh"
#include "Utils/Stopwatch.cuh"

#include "Configs.cuh"
#include "Parallelizer.cuh"
#include "Results.cuh"

void print_info(
    int& n_blocks,
    int& n_threads,
    ParametersManager& pm,
    int& n_vars,
    int& n_clauses,
    FormulaData& fdata,
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
        printf("Formula has %d vars and %d clauses\n", n_vars,
            n_clauses);
        printf("Variable '%d', the most frequent, has been found %d times.\n",
            fdata.get_most_common_var() + 1,
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
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
        printf("dynamically");
#else
        printf("statically");
#endif
        printf(" allocated vector.\n");
        printf("Formula clauses are stored in ");
#ifdef FORMULA_STORED_IN_SEVERAL_ALLOCATIONS
        printf("several allocations.\n");
#else
        printf("one allocation.\n");
#endif
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

int main(int argc, char* argv[])
{
    check(cudaSetDeviceFlags(cudaDeviceMapHost),
        "Setting device flag");

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

    if (fdata.get_n_vars() < MIN_VARIABLES_TO_PARALLELIZE && (pm.get_n_blocks() > 1 || pm.get_n_threads() > 1 || pm.get_sequential_as_parallel())) {
        printf("Warning: There are %d vars in the formula and at least %d "
               "are necessary to parallelize. Forcing sequential execution!\n",
            fdata.get_n_vars(), MIN_VARIABLES_TO_PARALLELIZE);
        pm.force_sequential_configuration();
    }

    const CUDAClauseVec& formula = fdata.get_formula_dev(); // Use reference
    const std::vector<Clause>& formula_host = *(fdata.get_formula_host()); // Use reference
    n_vars = fdata.get_n_vars();
    n_clauses = fdata.get_n_clauses();

    int n_threads = pm.get_n_threads();
    int n_blocks = pm.get_n_blocks();

    // This holds the max implication a var may have, to set the capacity of edges.
    int max_implication_per_var = std::max(fdata.get_largest_clause_size(), MIN_IMPLICATION_PER_VAR);

    print_info(
        n_blocks,
        n_threads,
        pm,
        n_vars,
        n_clauses,
        fdata,
        max_implication_per_var);

    if (fdata.get_status_after_preprocessing() != sat_status::UNDEF) {
        if (pm.get_verbosity_level() >= 1) {
            printf("Solved in pre-processing.\n");
        }

        Results res(n_vars, false);

        res.set_host_status(fdata.get_status_after_preprocessing());
        res.print_results(fdata.get_solved_literals(), formula_host);

        // No explicit fdata.cleanup() needed, destructor handles formula_dev
        return 0;
    }

    std::vector<Lit> const& solved_lits = fdata.get_solved_literals();
    // Initialize dead_vars_dev with correct capacity
    GPUVec<Var> dead_vars_dev(solved_lits.size());
    std::vector<Var> dead_vars_host;
    dead_vars_host.reserve(solved_lits.size()); // Reserve host vector capacity

    for (Lit lit : solved_lits) {
        Var v = var(lit);
        dead_vars_dev.add(v); // Add to device vector
        dead_vars_host.push_back(v); // Add to host vector
    }

    Results* results = nullptr; // Initialize results pointer

    int* state_ptr;
    KernelDataPod* data_ptr; // Pointer to buffer for KernelDataPod structs

    check(cudaMallocManaged(&state_ptr, sizeof(int)),
        "Alloc unified memory for state");
    *state_ptr = INT_MAX;

    // Allocate buffer for KernelDataPod structs
    check(cudaMallocManaged(&data_ptr, (sizeof(KernelDataPod) * n_blocks * n_threads)),
        "Alloc unified memory for KernelDataPod buffer");

    check(cudaDeviceSetLimit(cudaLimitStackSize, DEVICE_THREAD_STACK_LIMIT),
        "Set stack limit");

    check(cudaDeviceSetLimit(cudaLimitMallocHeapSize, DEVICE_THREAD_HEAP_LIMIT),
        "Set heap limit");

    float elapsedTime = 0.;
    CudaStopwatch stopwatch;

    // Create CUDA stream for asynchronous execution
    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "Create CUDA stream");

    // Declare pointers outside the if/else block
    DataToDevice* host_data_template_ptr = nullptr;
    JobChooser* chooser_ptr = nullptr; // Use base class pointer
    RuntimeStatistics* stat_ptr = nullptr; // Declare stat_ptr here, initialize to nullptr

    if (n_threads == 1 && n_blocks == 1 && !pm.get_sequential_as_parallel()) {
        // --- Sequential Path ---
        if (pm.get_verbosity_level() >= 1) {
            printf("Number of jobs = 1\n");
        }
#ifdef DEBUG
        printf("About to call sequential kernel!\n");
#endif
        DataToDevice::numbers n = {
            n_vars,
            n_clauses,
            0, // n_jobs
            1, // n_blocks
            1, // n_threads
            max_implication_per_var
        };

        DataToDevice::atomics counters = { nullptr, nullptr }; // No atomic counters needed

        // Allocate statistics object in managed memory for sequential run
        RuntimeStatistics* seq_stat_ptr = nullptr;
        check(cudaMallocManaged(&seq_stat_ptr, sizeof(RuntimeStatistics)), "Alloc managed memory for sequential RuntimeStatistics");
        new (seq_stat_ptr) RuntimeStatistics(n.blocks, n.threads, nullptr); // Construct using placement new

        // Create DataToDevice for sequential run, passing the pointer
        DataToDevice seq_data(formula, dead_vars_dev,
            seq_stat_ptr, // Pass the pointer
            n, counters, 0, 0); // Pass 0 for job/literal capacity

        seq_data.prepare_sequencial(); // Prepare specific sequential data if needed

        stopwatch.start();
        run_sequential<<<1, 1, 0, stream>>>(seq_data, state_ptr);
        elapsedTime = stopwatch.stop();

        results = seq_data.get_results_ptr();

        // Ensure cleanup for sequential data object
        seq_data.cleanup();
        // Cleanup the managed statistics object for sequential path
        if (seq_stat_ptr) {
            seq_stat_ptr->~RuntimeStatistics(); // Explicit destructor call
            check(cudaFree(seq_stat_ptr), "Free managed sequential RuntimeStatistics");
        }
    } else {
        // Declare device object pointers here, before the parallel block, initialize to null
        JobsQueue* d_jobs_queue_ptr = nullptr;
        CUDAClauseVec* d_clauses_db_obj_ptr = nullptr;
        Results* d_results_ptr = nullptr; // Pointer for device Results object
        sat_status* d_results_array_ptr = nullptr; // Pointer for device results array
        sat_status* d_formula_status_ptr = nullptr; // Pointer for device formula status

        // --- Parallel Path ---
        // --- Parallel Path ---

        // Allocate the SimpleJobChooser (only supported type for now)
        chooser_ptr = new SimpleJobChooser(n_vars, dead_vars_host);

        // Evaluate chooser to determine job structure
        chooser_ptr->evaluate(); // Use pointer

        // Calculate required capacities *before* creating DataToDevice
        size_t job_capacity = chooser_ptr->get_n_jobs(); // Use pointer
        size_t literal_capacity = chooser_ptr->estimateTotalLiteralSize(); // Use pointer

        if (pm.get_verbosity_level() >= 1) {
            printf("Number of jobs = %llu\n", job_capacity);
            printf("Total literals across jobs = %llu\n", literal_capacity);
        }

        // Setup atomic counters for parallel execution
        DataToDevice::atomics atomics;
        unsigned zero = 0;
        check(cudaMalloc(&atomics.next_job, sizeof(unsigned)), "Allocating next_job counter");
        check(cudaMemcpy(atomics.next_job, &zero, sizeof(unsigned), cudaMemcpyHostToDevice),
            "Zeroing next_job counter");
        check(cudaMalloc(&atomics.completed_jobs, sizeof(unsigned)), "Allocating completed_jobs counter");
        check(cudaMemcpy(atomics.completed_jobs, &zero, sizeof(unsigned), cudaMemcpyHostToDevice),
            "Zeroing completed_jobs counter");

        // Setup numbers struct
        DataToDevice::numbers n = {
            n_vars,
            n_clauses,
            (int)job_capacity, // Cast size_t to int if needed by struct
            n_blocks,
            n_threads,
            max_implication_per_var
        };

        // Allocate statistics object in managed memory

        RuntimeStatistics* stat_ptr = nullptr;
        check(cudaMallocManaged(&stat_ptr, sizeof(RuntimeStatistics)), "Alloc managed memory for RuntimeStatistics");
        // Use placement new to construct the object in the allocated memory
        new (stat_ptr) RuntimeStatistics(n.blocks, n.threads, atomics.completed_jobs);

        // Create the single host-side DataToDevice instance using the calculated capacities
        // Pass the pointer to the managed RuntimeStatistics object
        host_data_template_ptr = new DataToDevice(formula, dead_vars_dev, stat_ptr, n, atomics,
            job_capacity, literal_capacity);

        // Prepare the parallel data (populates the JobsQueue)
        host_data_template_ptr->prepare_parallel(*chooser_ptr // Use pointer
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
                                                              // , assumptions // Pass assumptions if needed by prepare_parallel
#endif
        );

        // --- Allocate and copy CUDAClauseVec object to device ---
        d_clauses_db_obj_ptr = thrust::raw_pointer_cast(thrust::device_malloc<CUDAClauseVec>(1));

        // Assuming get_clauses_db() returns a const reference to the host object
        const CUDAClauseVec& h_clauses_db_obj = host_data_template_ptr->get_clauses_db();
        check(cudaMemcpy(d_clauses_db_obj_ptr, &h_clauses_db_obj, sizeof(CUDAClauseVec), cudaMemcpyHostToDevice), "Copying CUDAClauseVec object to device");
        // --- End allocation and copy ---
        // --- Allocate and copy JobsQueue object to device ---

        d_jobs_queue_ptr = thrust::raw_pointer_cast(thrust::device_malloc<JobsQueue>(1));

        // Get a reference to the host JobsQueue object (it contains the correct device pointer for next_job_index)
        JobsQueue& h_jobs_queue_obj = host_data_template_ptr->get_jobs_queue();
        check(cudaMemcpy(d_jobs_queue_ptr, &h_jobs_queue_obj, sizeof(JobsQueue), cudaMemcpyHostToDevice), "Copying JobsQueue object to device");
        // --- End allocation and copy ---

        // --- Allocate and prepare Results object on device ---
        d_results_ptr = thrust::raw_pointer_cast(thrust::device_malloc<Results>(1)); // Allocate device memory for Results object
        d_results_array_ptr = thrust::raw_pointer_cast(thrust::device_malloc<sat_status>(n_vars)); // Allocate device memory for results array
        d_formula_status_ptr = thrust::raw_pointer_cast(thrust::device_malloc<sat_status>(1)); // Allocate device memory for formula status

        // Initialize device formula status to UNDEF
        sat_status initial_status = sat_status::UNDEF;
        check(cudaMemcpy(d_formula_status_ptr, &initial_status, sizeof(sat_status), cudaMemcpyHostToDevice), "Initializing device formula status");

        // Create a temporary host object to act as a template
        // IMPORTANT: This assumes the Results constructor doesn't do incompatible host/device things.
        Results h_results_template(n_vars, true); // Create on stack, constructor sets on_gpu=true

        // Use the new public method to set the device pointers and n_vars
        h_results_template.init_device_pointers(d_results_array_ptr, d_formula_status_ptr, n_vars);

        // Copy the prepared host template to the allocated device memory
        check(cudaMemcpy(d_results_ptr, &h_results_template, sizeof(Results), cudaMemcpyHostToDevice), "Copying prepared Results template to device");
        // --- End Results object preparation ---

        // Populate the KernelDataPod buffer for each thread/block context
        KernelDataPod* data_iter = data_ptr;
        // Get literal buffer base pointer *from the host object* before the loop, as it points to device memory managed by GPUVec
        Lit* literal_base_ptr = host_data_template_ptr->get_jobs_queue().get_literal_buffer_ptr(); // Get base pointer once

        for (size_t i = 0; i < (size_t)n_blocks * n_threads; ++i) {
            KernelDataPod pod;
            // Get the pointer from DataToDevice (which now holds the managed pointer)
            pod.statistics_ptr = host_data_template_ptr->get_statistics_ptr();
            // Assign pointer to the DEVICE copy of the JobsQueue object
            pod.queue_ptr = d_jobs_queue_ptr;
            // Assign pointer to the DEVICE copy of the CUDAClauseVec object
            pod.clauses_db_ptr = d_clauses_db_obj_ptr;
            // Get raw pointer and size from the GPUVec object
            GPUVec<Var>* dead_vars_gpuvec_ptr = host_data_template_ptr->get_dead_vars_ptr();
            pod.dead_vars_elements_ptr = dead_vars_gpuvec_ptr ? dead_vars_gpuvec_ptr->data() : nullptr; // Use data() method
            pod.dead_vars_size = dead_vars_gpuvec_ptr ? dead_vars_gpuvec_ptr->size_of() : 0;
            pod.nodes_repository_ptr = host_data_template_ptr->get_nodes_repository_ptr();
            pod.found_answer_ptr = host_data_template_ptr->get_found_answer_ptr();
            // pod.results_ptr = host_data_template_ptr->get_results_ptr(); // OLD: Host pointer
            pod.results_ptr = d_results_ptr; // NEW: Device pointer
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
            // TODO: Need a getter for all_assumptions_parallel_ptr if required by kernel
            // pod.all_assumptions_parallel_ptr = host_data_template_ptr->get_all_assumptions_parallel_ptr();
#endif
            pod.n_vars = host_data_template_ptr->get_n_vars();
            pod.max_implication_per_var = host_data_template_ptr->get_max_implication_per_var();
            pod.literal_buffer_base_ptr = literal_base_ptr; // Populate the new pointer

            memcpy(data_iter, &pod, sizeof(KernelDataPod));
            data_iter++;
        }

        printf("Synchronizing after KernelDataPod memcpy loop...\n");
        cudaDeviceSynchronize();
        check(cudaGetLastError(), "Error after KernelDataPod memcpy loop sync");
        printf("Sync after memcpy loop successful.\n");

        KernelContextStorage thread_contexts;
        check(cudaMallocPitch(&thread_contexts.data,
                  &thread_contexts.pitch, n_threads * sizeof(void*), n_blocks),
            "Allocate thread contexts");
        printf("cudaMallocPitch Result: data_ptr = %p, pitch = %zu\n", thread_contexts.data, thread_contexts.pitch);
        cudaDeviceSynchronize();
        check(cudaGetLastError(), "Error after cudaMallocPitch sync");

        printf("About to invoke kernel...\n");
        printf("Kernel Launch Params: n_blocks = %d, n_threads = %d\n", n_blocks, n_threads);
        fflush(stdout);

        cudaDeviceSynchronize();
        check(cudaGetLastError(), "Error before parallel_kernel_init launch");

        parallel_kernel_init<<<n_blocks, n_threads, 0, stream>>>(data_ptr, thread_contexts);
        check(cudaGetLastError(), "Error during parallel_kernel_init launch configuration");
        cudaDeviceSynchronize();
        check(cudaGetLastError(), "Error during parallel_kernel_init execution");
        printf("parallel_kernel_init finished.\n");

        size_t call = 1;
        stopwatch.start();
        while (true) {
            parallel_kernel<<<n_blocks, n_threads, 0, stream>>>(thread_contexts, state_ptr);
            check(cudaGetLastError(), "Error during parallel_kernel launch configuration");

            // Check state periodically, synchronizing before check
            // Adjust frequency as needed (e.g., every N calls)
            if (call % 10 == 0) { // Example: check every 10 calls
                cudaDeviceSynchronize();
                check(cudaGetLastError(), "Error checking state_ptr");
                if (*state_ptr != INT_MAX)
                    break;
            }

            if (call == SIZE_MAX) {
                cudaDeviceSynchronize(); // Sync before breaking
                break;
            }
            call++;
        }
        // Final sync to ensure state is up-to-date before stopping timer
        cudaDeviceSynchronize();
        check(cudaGetLastError(), "Error after parallel_kernel loop sync");
        elapsedTime = stopwatch.stop();

        printf("Kernel was invoked %zu times\n", (call)); // Corrected count

        if (call == SIZE_MAX) {
            printf("Max invocations reached\n");
            // Perform cleanup before returning error
            delete chooser_ptr; // Delete chooser
            if (host_data_template_ptr)
                host_data_template_ptr->cleanup();
            delete host_data_template_ptr;
            // Free managed stat_ptr
            if (stat_ptr) {
                stat_ptr->~RuntimeStatistics(); // Explicit destructor call
                check(cudaFree(stat_ptr), "Free managed RuntimeStatistics");
                stat_ptr = nullptr; // Avoid double free later
            }
            check(cudaFree(atomics.next_job), "Free next_job counter");
            check(cudaFree(atomics.completed_jobs), "Free completed_jobs counter");
            check(cudaFree(state_ptr), "Free state memory");
            check(cudaFree(data_ptr), "Free data memory");
            check(cudaFree(thread_contexts.data), "Free thread contexts");
            // No explicit fdata cleanup
            dead_vars_dev.destroy();
            cudaStreamDestroy(stream);
            cudaDeviceReset();
            return -1;
        }

        KernelDataPod* winner_pod = &data_ptr[*state_ptr];

        // Removed call to parallel_kernel_retrieve_results
        // The results are now set directly by the winning thread in parallel_kernel

        // results = winner_pod->results_ptr; // OLD: This is the DEVICE pointer

        // --- Copy results back from Device Results object to Host ---
        Results final_host_results(n_vars, false); // Create final host object
        sat_status final_status_host = sat_status::UNDEF; // Host variable for status

        // Copy status back from device memory pointed to by d_formula_status_ptr
        check(cudaMemcpy(&final_status_host, d_formula_status_ptr, sizeof(sat_status), cudaMemcpyDeviceToHost), "Copying final status from device");
        final_host_results.set_host_status(final_status_host); // Set status in host object

        // If SAT, copy the results array back (print_results needs it)
        // Note: print_results currently takes solved_literals and formula,
        // it might need adjustment to work with the sat_status array directly,
        // or we need to reconstruct solved_literals from the status array.
        // For now, we just set the status. The print call later might fail or show wrong results.
        // TODO: Adjust result printing logic if needed based on final_host_results.

        results = &final_host_results; // Point 'results' to the new host object for subsequent code

#ifdef ENABLE_STATISTICS
        RuntimeStatistics* winner_stats = winner_pod->statistics_ptr;
        // Check if winner_stats is valid before using it
        if (winner_stats) {
            int solved_jobs = winner_stats->get_all_threads_total_completed_jobs();
            printf("Jobs capacity = %zu\n", job_capacity);
            printf("There were %d solved jobs reported by winner statistics.\n", solved_jobs);
            cudaDeviceSynchronize();
            // winner_stats->print_function_time_statistics();
        } else {
            printf("Winner statistics pointer is null.\n");
        }
#endif // ENABLE_STATISTICS

        check(cudaFree(atomics.next_job), "Free next_job counter");
        check(cudaFree(atomics.completed_jobs), "Free completed_jobs counter");
        check(cudaFree(thread_contexts.data), "Free thread contexts");
        // Free managed stat_ptr
        if (stat_ptr) {
            stat_ptr->~RuntimeStatistics(); // Explicit destructor call
            check(cudaFree(stat_ptr), "Free managed RuntimeStatistics");
            stat_ptr = nullptr; // Avoid double free later
        }
        if (d_jobs_queue_ptr) { // Check if allocated (should be in parallel path)
            check(cudaFree(d_jobs_queue_ptr), "Free device JobsQueue object");
        }
        if (d_clauses_db_obj_ptr) { // Check if allocated
            check(cudaFree(d_clauses_db_obj_ptr), "Free device CUDAClauseVec object");
        }
        // Free device Results object and its internal arrays
        if (d_results_ptr) {
            check(cudaFree(d_results_ptr), "Free device Results object");
        }
        if (d_results_array_ptr) {
            check(cudaFree(d_results_array_ptr), "Free device results array");
        }
        if (d_formula_status_ptr) {
            check(cudaFree(d_formula_status_ptr), "Free device formula status");
            d_jobs_queue_ptr = nullptr; // Prevent double free if cleanup logic is complex
        }
    } // End parallel path

    printf("Total time on GPU: %f ms\n", elapsedTime);

    if (results) {
        results->print_results(fdata.get_solved_literals(), formula_host);
    } else {
        printf("Error: Results pointer is null.\n");
    }

    if (pm.get_write_log()) {
        char buf[256];
        std::ofstream out("autolog.txt", std::ios_base::app);
        std::snprintf(buf, sizeof(buf), "%s,%d,%d,%f\n",
            pm.get_input_file(), n_threads, n_blocks, elapsedTime);
        out << buf;
    }

    // --- Final Cleanup ---
    delete chooser_ptr; // Delete chooser if allocated

    if (host_data_template_ptr) {
        host_data_template_ptr->cleanup();
        delete host_data_template_ptr;
        host_data_template_ptr = nullptr;
    }

    // Free managed stat_ptr if not freed in parallel path already
    if (stat_ptr) {
        stat_ptr->~RuntimeStatistics(); // Explicit destructor call
        check(cudaFree(stat_ptr), "Free managed RuntimeStatistics");
        stat_ptr = nullptr;
    }

    // Cleanup dead_vars_dev explicitly
    dead_vars_dev.destroy();

    // No explicit fdata cleanup needed

    check(cudaFree(state_ptr), "Free state memory");
    check(cudaFree(data_ptr), "Free data memory");

    check(cudaStreamDestroy(stream), "Destroy CUDA stream");

    cudaDeviceReset();

    return 0;
}
