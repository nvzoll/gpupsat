#include "RuntimeStatistics.cuh"

RuntimeStatistics::RuntimeStatistics(int n_blocks, int n_threads, unsigned *total_jobs)
    : n_blocks { n_blocks }
    , n_threads { n_threads }
    , all_threads_total_completed_jobs_dev { total_jobs }
{
    alloc_stats(n_blocks, n_threads);
    clear_stats(n_blocks, n_threads);
}

__device__ int RuntimeStatistics::get_index()
{
    //return threadIdx.x + blockIdx.x * blockDim.x;
    return threadIdx.x + (blockIdx.x * gridDim.y + blockIdx.y) * blockDim.x;
}

__device__ void RuntimeStatistics::signal_job_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_solving_time_dev, n_jobs_run_dev);
#endif
}

__device__ void RuntimeStatistics::signal_job_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_solving_time_dev);
#endif
}

__host__ std::vector<uint64_t> RuntimeStatistics::get_total_solving_time()
{
#ifdef ENABLE_STATISTICS
    std::vector<uint64_t> total_solving_time(n_blocks * n_threads);

    size_t size = sizeof(uint64_t) * total_solving_time.size();
    cudaMemcpy(total_solving_time.data(), total_solving_time_dev, size, cudaMemcpyDeviceToHost);

    return total_solving_time;
#else // ENABLE_STATISTICS
    return nullptr;
#endif // ENABLE_STATISTICS
}

__host__ __device__ int *RuntimeStatistics::get_n_jobs_run()
{
#ifdef ENABLE_STATISTICS
#ifndef __CUDA_ARCH__
    int size = sizeof(int) * n_blocks * n_threads;
    int *numb_of_jobs_run = new int[n_blocks * n_threads];
    cudaMemcpy(numb_of_jobs_run, n_jobs_run_dev, size, cudaMemcpyDeviceToHost);
    return numb_of_jobs_run;
#else // __CUDA_ARCH__
    return n_jobs_run_dev;
#endif // __CUDA_ARCH__

#else // ENABLE_STATISTICS
    return nullptr;
#endif // ENABLE_STATISTICS
}

__device__ void RuntimeStatistics::signal_preprocess_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_on_gpu_preprocessing_dev, n_preprocesses_made_dev);
#endif
}

__device__ void RuntimeStatistics::signal_preprocess_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_on_gpu_preprocessing_dev);
#endif
}

__device__ void RuntimeStatistics::signal_reset_structures_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_resetting_structures_dev, n_resets_done);
#endif
}

__device__ void RuntimeStatistics::signal_reset_structures_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_resetting_structures_dev);
#endif
}

__device__ void RuntimeStatistics::signal_conflict_analysis_start(int decision_level)
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    signal_start(total_time_analyzing_conflict_dev, n_conflict_analysis_made_dev);
    last_backtracked_start_level_dev[index] = decision_level;
#endif
}

__device__ void RuntimeStatistics::signal_conflict_analysis_stop(int decision_level)
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    signal_stop(total_time_analyzing_conflict_dev);


    if (decision_level < last_backtracked_start_level_dev[index]) {
        long int value = last_backtracked_start_level_dev[index] - decision_level;
        total_backtracked_levels_dev[index] += value;
        backtracked_levels_instances_count_dev[index]++;
    }
#endif
}

__device__ void RuntimeStatistics::signal_decision_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_deciding_dev, n_decisions_made_dev);
#endif
}

__device__ void RuntimeStatistics::signal_decision_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_deciding_dev);
#endif
}

__device__ void RuntimeStatistics::signal_backtrack_start(int n_decisions)
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_backtracking_dev, n_backtracks_made_dev);
    int index = get_index();

    sum_of_decision_before_backtracking_dev[index] += n_decisions;
#endif
}

__device__ void RuntimeStatistics::signal_backtrack_stop(int n_decisions)
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_backtracking_dev);
    int index = get_index();
    sum_of_decision_after_backtracking_dev[index] += n_decisions;
#endif
}

__device__ void RuntimeStatistics::signal_create_structures_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_creating_structures_dev, n_create_structures_done);
#endif
}

__device__ void RuntimeStatistics::signal_create_structures_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_creating_structures_dev);
#endif
}

__device__ void RuntimeStatistics::signal_next_job_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_on_next_job_dev, n_next_job_done);
#endif
}

__device__ void RuntimeStatistics::signal_next_job_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_on_next_job_dev);
#endif
}

__device__ void RuntimeStatistics::signal_add_jobs_to_assumptions_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_on_add_jobs_to_assumptions_dev, n_add_jobs_to_assumptions_done);
#endif
}

__device__ void RuntimeStatistics::signal_add_jobs_to_assumptions_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_on_add_jobs_to_assumptions_dev);
#endif
}

__device__ void RuntimeStatistics::signal_process_results_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_processing_results_dev, n_process_results_done);
#endif
}

__device__ void RuntimeStatistics::signal_process_results_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_processing_results_dev);
#endif
}

__device__ void RuntimeStatistics::signal_pre_proc_handle_assumptions_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_on_gpu_preprocessing_handling_assumption_dev, n_preprocesses_handling_assumption_made_dev);
#endif
}

__device__ void RuntimeStatistics::signal_pre_proc_handle_assumptions_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_on_gpu_preprocessing_handling_assumption_dev);
#endif
}

__device__ void RuntimeStatistics::signal_pre_proc_add_to_graph_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_on_gpu_preprocessing_adding_to_graph_dev, n_preprocesses_adding_to_graph_made_dev);
#endif
}

__device__ void RuntimeStatistics::signal_pre_proc_add_to_graph_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_on_gpu_preprocessing_adding_to_graph_dev);
#endif
}

__device__ void RuntimeStatistics::signal_pre_proc_handling_vars_start()
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time_on_gpu_preprocessing_handling_vars_dev, n_preprocesses_handling_vars_made_dev);
#endif
}

__device__ void RuntimeStatistics::signal_pre_proc_handling_vars_stop()
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time_on_gpu_preprocessing_handling_vars_dev);
#endif
}

__device__ void RuntimeStatistics::signal_start(uint64_t *total_time,
        int *occurrences)
{
#ifdef ENABLE_STATISTICS
    signal_start(total_time, occurrences, clock64());
#endif
}

__device__ void RuntimeStatistics::signal_start(uint64_t *total_time, int *occurrences, uint64_t time)
{
#ifdef ENABLE_STATISTICS
    int index = get_index();

    if (occurrences[index] == 0) {
        total_time[index] = -time;    //-(((long int) clock64()) >> STATISTICS_TIME_OFFSET);
    }
    else {
        total_time[index] -= time;    //((long int) clock64()) >> STATISTICS_TIME_OFFSET;
    }

    occurrences[index]++;
#endif
}

__device__ void RuntimeStatistics::signal_stop(uint64_t *total_time)
{
#ifdef ENABLE_STATISTICS
    signal_stop(total_time, clock64());
#endif
}

__device__ void RuntimeStatistics::signal_stop(uint64_t *total_time, uint64_t time)
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    total_time[index] += time;//((long int) clock64()) >> STATISTICS_TIME_OFFSET;
#endif
}

template<class T>
__host__ std::vector<T> RuntimeStatistics::copy_to_host(bool block_thread_vector, T *element_dev)
{
    int size = (block_thread_vector) ? n_blocks * n_threads : 1;

    std::vector<T> element_host(size);

    check(cudaMemcpy(element_host.data(), element_dev,
                     size * sizeof(T), cudaMemcpyDeviceToHost),
                     "Copying statistics data from device");

    return element_host;
}

__host__ void RuntimeStatistics::print_function_time_statistics()
{
#ifdef ENABLE_STATISTICS
    printf("\n*****Statistics******\n");
    printf("Total job's time:\n");
    print_stat_average(total_solving_time_dev, n_jobs_run_dev);
    print_stat_total(total_solving_time_dev, n_jobs_run_dev);

    printf("Pre-processing time:\n");
    print_stat_average(total_time_on_gpu_preprocessing_dev, n_preprocesses_made_dev);
    print_stat_total(total_time_on_gpu_preprocessing_dev, n_preprocesses_made_dev);

    printf("Decision time:\n");
    print_stat_average(total_time_deciding_dev, n_decisions_made_dev);
    print_stat_total(total_time_deciding_dev, n_decisions_made_dev);

    printf("Conflict analyzing time:\n");
    print_stat_average(total_time_analyzing_conflict_dev, n_conflict_analysis_made_dev);
    print_stat_total(total_time_analyzing_conflict_dev, n_conflict_analysis_made_dev);

    printf("Backtracking time:\n");
    print_stat_average(total_time_backtracking_dev, n_backtracks_made_dev);
    print_stat_total(total_time_backtracking_dev, n_backtracks_made_dev);

    printf("Structures reset time:\n");
    print_stat_average(total_time_resetting_structures_dev, n_resets_done);
    print_stat_total(total_time_resetting_structures_dev, n_resets_done);

    printf("Creating structures time:\n");
    print_stat_average(total_time_creating_structures_dev, n_create_structures_done);
    print_stat_total(total_time_creating_structures_dev, n_create_structures_done);

    printf("Next job time:\n");
    print_stat_average(total_time_on_next_job_dev, n_next_job_done);
    print_stat_total(total_time_on_next_job_dev, n_next_job_done);

    printf("Add jobs to assumptions time:\n");
    print_stat_average(total_time_on_add_jobs_to_assumptions_dev, n_add_jobs_to_assumptions_done);
    print_stat_total(total_time_on_add_jobs_to_assumptions_dev, n_add_jobs_to_assumptions_done);

    printf("Processing results time:\n");
    print_stat_average(total_time_processing_results_dev, n_process_results_done);
    print_stat_total(total_time_processing_results_dev, n_process_results_done);

    printf("Average backtracked levels:\n");
    print_stat_average(total_backtracked_levels_dev, backtracked_levels_instances_count_dev, false);

    printf("Pre-processing - handling assumptions time:\n");
    print_stat_average(total_time_on_gpu_preprocessing_handling_assumption_dev,
                       n_preprocesses_handling_assumption_made_dev);
    print_stat_total(total_time_on_gpu_preprocessing_handling_assumption_dev, n_preprocesses_handling_assumption_made_dev);

    printf("Pre-processing - adding assumptions to graph time:\n");
    print_stat_average(total_time_on_gpu_preprocessing_adding_to_graph_dev, n_preprocesses_adding_to_graph_made_dev);
    print_stat_total(total_time_on_gpu_preprocessing_adding_to_graph_dev, n_preprocesses_adding_to_graph_made_dev);

    printf("Pre-processing - adding handling vars time:\n");
    print_stat_average(total_time_on_gpu_preprocessing_handling_vars_dev, n_preprocesses_handling_vars_made_dev);
    print_stat_total(total_time_on_gpu_preprocessing_handling_vars_dev, n_preprocesses_handling_vars_made_dev);
#endif // ENABLE_STATISTICS
}

__host__ void RuntimeStatistics::print_stat_average(uint64_t *block_thread_stats_dev, int *count_dev,
        bool use_offset)
{
#ifdef ENABLE_STATISTICS
    std::vector<uint64_t> stats_host = copy_to_host<uint64_t>(true, block_thread_stats_dev);
    std::vector<int> count_host = copy_to_host<int>(true, count_dev);

    printf("Average:\n");

    for (int block = 0; block < n_blocks; block++) {
        for (int thread = 0; thread < n_threads; thread++) {
            int index = block * n_threads + thread;

            printf("\tBlock %d, thread %d: ", block, thread);

            if (count_host[index] == 0) {
                printf("not run\n");
            }
            else {
                if (use_offset) {
                    printf("%15.3f\n", ((double)(stats_host[index] << STATISTICS_TIME_OFFSET) / count_host[index]));
                }
                else {
                    printf("%15.3f\n", ((double)(stats_host[index]) / count_host[index]));
                }
            }
        }
    }
#endif // ENABLE_STATISTICS
}

__host__ void RuntimeStatistics::print_stat_total(uint64_t *block_thread_stats_dev, int *count_dev)
{
#ifdef ENABLE_STATISTICS
    std::vector<uint64_t> stats_host = copy_to_host<uint64_t>(true, block_thread_stats_dev);
    std::vector<int> count_host = copy_to_host<int>(true, count_dev);

    printf("Total:\n");

    for (int block = 0; block < n_blocks; block++) {
        for (int thread = 0; thread < n_threads; thread++) {
            int index = block * n_threads + thread;

            printf("\tBlock %d, thread %d: ", block, thread);

            if (count_host[index] == 0) {
                printf("not run\n");
            }
            else {
                printf("%zu, run %d times\n", (stats_host[index] << STATISTICS_TIME_OFFSET) , count_host[index]);
            }
        }
    }
#endif // ENABLE_STATISTICS
}

template<class T>
__host__ void RuntimeStatistics::print_stat_sum(T *block_thread_stats_dev)
{

}

__host__ void RuntimeStatistics::print_average()
{
#ifdef ENABLE_STATISTICS
    std::vector<uint64_t> time = get_total_solving_time();
    int *count = get_n_jobs_run();

    for (int i = 0; i < n_blocks; i++) {
        for (int j = 0; j < n_threads; j++) {
            int index = j + i * n_threads;
            uint64_t average = time[index] / count[index];

            printf("Block %d, thread %d has average job solving time = %zu\n", i, j, average);
        }
    }

    delete[] count;
#endif // ENABLE_STATISTICS
}

__device__ void RuntimeStatistics::add_completed_jobs(int completed_jobs)
{
#ifdef ENABLE_STATISTICS
    atomicAdd(all_threads_total_completed_jobs_dev, completed_jobs);
#endif
}

__host__ __device__ int RuntimeStatistics::get_all_threads_total_completed_jobs()
{
#ifdef ENABLE_STATISTICS
#ifndef __CUDA_ARCH__
    int value;
    cudaMemcpy(&value, all_threads_total_completed_jobs_dev, sizeof(int), cudaMemcpyDeviceToHost);
    return value;
#else // __CUDA_ARCH__
    return *all_threads_total_completed_jobs_dev;
#endif // __CUDA_ARCH__

#else // ENABLE_STATISTICS
    return 0;
#endif
}

__device__ uint64_t RuntimeStatistics::get_total_time_solving()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return total_solving_time_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}

__device__ uint64_t RuntimeStatistics::get_total_deciding_time()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return total_time_deciding_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}

__device__ uint64_t RuntimeStatistics::get_total_backtracking_time()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return total_time_backtracking_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}

__device__ uint64_t RuntimeStatistics::get_total_conflict_analyzing_time()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return total_time_analyzing_conflict_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}

__device__ int RuntimeStatistics::get_sum_of_decision_before_backtracking()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return sum_of_decision_before_backtracking_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}
__device__ int RuntimeStatistics::get_sum_of_decision_after_backtracking()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return sum_of_decision_after_backtracking_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}
__device__ uint64_t RuntimeStatistics::get_total_backtracked_levels()
{
#ifdef ENABLE_STATISTICS
    int index = get_index();
    return total_backtracked_levels_dev[index];
#else
    return 0;
#endif // ENABLE_STATISTICS
}

template<class T>
__host__ void clear_stat(T *&element, bool block_thread_vector, int n_blocks, int n_threads)
{
#ifdef ENABLE_STATISTICS
    size_t size = (block_thread_vector) ? n_blocks * n_threads : 1;

    std::vector<T> host_temp(size);

    check(cudaMemcpy(element, host_temp.data(), size * sizeof(T),
                     cudaMemcpyHostToDevice), "Clearing statistics on device");
#endif // ENABLE_STATISTICS
}

__host__ void RuntimeStatistics::clear_stats(int n_blocks, int n_threads)
{
#ifdef ENABLE_STATISTICS
    clear_stat<uint64_t>(total_solving_time_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_jobs_run_dev, true, n_blocks, n_threads);
    // Partial time
    clear_stat<uint64_t>(total_time_analyzing_conflict_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_conflict_analysis_made_dev, true, n_blocks, n_threads);
    clear_stat<uint64_t>(total_time_deciding_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_decisions_made_dev, true, n_blocks, n_threads);
    clear_stat<uint64_t>(total_time_backtracking_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_backtracks_made_dev, true, n_blocks, n_threads);
    clear_stat<uint64_t>(total_time_on_gpu_preprocessing_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_preprocesses_made_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_resets_done, true, n_blocks, n_threads);

    clear_stat<int>(n_create_structures_done, true, n_blocks, n_threads);
    clear_stat<int>(n_next_job_done, true, n_blocks, n_threads);
    clear_stat<int>(n_add_jobs_to_assumptions_done, true, n_blocks, n_threads);
    clear_stat<int>(n_process_results_done, true, n_blocks, n_threads);

    // Decisions
    clear_stat<int>(sum_of_decision_before_backtracking_dev, true, n_blocks, n_threads);
    clear_stat<int>(sum_of_decision_after_backtracking_dev, true, n_blocks, n_threads);
    clear_stat<uint64_t>(total_backtracked_levels_dev, true, n_blocks, n_threads);
    clear_stat<int>(last_backtracked_start_level_dev, true, n_blocks, n_threads);
    clear_stat<int>(backtracked_levels_instances_count_dev, true, n_blocks, n_threads);

    // Pre-processing
    clear_stat<uint64_t>(total_time_on_gpu_preprocessing_handling_assumption_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_preprocesses_handling_assumption_made_dev, true, n_blocks, n_threads);
    clear_stat<uint64_t>(total_time_on_gpu_preprocessing_adding_to_graph_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_preprocesses_adding_to_graph_made_dev, true, n_blocks, n_threads);
    clear_stat<uint64_t>(total_time_on_gpu_preprocessing_handling_vars_dev, true, n_blocks, n_threads);
    clear_stat<int>(n_preprocesses_handling_vars_made_dev, true, n_blocks, n_threads);
#endif // ENABLE_STATISTICS
}

template<class T>
__host__ void alloc_stat(T *&element, bool block_thread_vector,
                         int n_blocks, int n_threads)
{
#ifdef ENABLE_STATISTICS
    size_t size = sizeof(T);

    if (block_thread_vector) {
        size *= n_blocks * n_threads;
    }

    check(cudaMalloc(&element, size), "Allocating structure for statistics");
#endif
}

__host__ void RuntimeStatistics::alloc_stats(int n_blocks, int n_threads)
{
#ifdef ENABLE_STATISTICS
    alloc_stat<uint64_t>(total_solving_time_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_jobs_run_dev, true, n_blocks, n_threads);
    // Partial time
    alloc_stat<uint64_t>(total_time_analyzing_conflict_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_conflict_analysis_made_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_deciding_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_decisions_made_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_backtracking_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_backtracks_made_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_on_gpu_preprocessing_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_preprocesses_made_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_resetting_structures_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_resets_done, true, n_blocks, n_threads);

    alloc_stat<uint64_t>(total_time_creating_structures_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_create_structures_done, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_on_next_job_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_next_job_done, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_on_add_jobs_to_assumptions_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_add_jobs_to_assumptions_done, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_processing_results_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_process_results_done, true, n_blocks, n_threads);

    // Decisions
    alloc_stat<int>(sum_of_decision_before_backtracking_dev, true, n_blocks, n_threads);
    alloc_stat<int>(sum_of_decision_after_backtracking_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_backtracked_levels_dev, true, n_blocks, n_threads);
    alloc_stat<int>(last_backtracked_start_level_dev, true, n_blocks, n_threads);
    alloc_stat<int>(backtracked_levels_instances_count_dev, true, n_blocks, n_threads);
    // Statistics from checking correction

    // Pre-processing
    alloc_stat<uint64_t>(total_time_on_gpu_preprocessing_handling_assumption_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_preprocesses_handling_assumption_made_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_on_gpu_preprocessing_adding_to_graph_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_preprocesses_adding_to_graph_made_dev, true, n_blocks, n_threads);
    alloc_stat<uint64_t>(total_time_on_gpu_preprocessing_handling_vars_dev, true, n_blocks, n_threads);
    alloc_stat<int>(n_preprocesses_handling_vars_made_dev, true, n_blocks, n_threads);
#endif // ENABLE_STATISTICS
}
