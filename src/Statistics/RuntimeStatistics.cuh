#ifndef __RUNTIMESTATISTICS_CUH__
#define __RUNTIMESTATISTICS_CUH__

#include <stdio.h>
#include <stdint.h>
#include <vector>

#include "SATSolver/Configs.cuh"
#include "StatisticsCalculator.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"

class RuntimeStatistics
{
public:
    RuntimeStatistics(int n_blocks, int n_threads, unsigned *total_jobs);
    __device__ void add_completed_jobs(int completed_jobs);
    __device__ void signal_job_start();
    __device__ void signal_job_stop();
    __device__ void signal_preprocess_start();
    __device__ void signal_preprocess_stop();
    __device__ void signal_conflict_analysis_start(int decision_level);
    __device__ void signal_conflict_analysis_stop(int decision_level);
    __device__ void signal_decision_start();
    __device__ void signal_decision_stop();
    __device__ void signal_backtrack_start(int n_decisions);
    __device__ void signal_backtrack_stop(int n_decisions);
    __device__ void signal_reset_structures_start();
    __device__ void signal_reset_structures_stop();

    /*
    starting structures
    next jobs
    adding jobs to assumptions
    processing results
    */
    __device__ void signal_create_structures_start();
    __device__ void signal_create_structures_stop();
    __device__ void signal_next_job_start();
    __device__ void signal_next_job_stop();
    __device__ void signal_add_jobs_to_assumptions_start();
    __device__ void signal_add_jobs_to_assumptions_stop();
    __device__ void signal_process_results_start();
    __device__ void signal_process_results_stop();
    __device__ void signal_pre_proc_handle_assumptions_start();
    __device__ void signal_pre_proc_handle_assumptions_stop();
    __device__ void signal_pre_proc_add_to_graph_start();
    __device__ void signal_pre_proc_add_to_graph_stop();
    __device__ void signal_pre_proc_handling_vars_start();
    __device__ void signal_pre_proc_handling_vars_stop();


    __host__ std::vector<uint64_t> get_total_solving_time();
    __host__ __device__ int *get_n_jobs_run();
    __host__ __device__ int get_all_threads_total_completed_jobs();
    __host__ void print_average();
    __host__  void print_function_time_statistics();


    __device__ uint64_t get_total_time_solving();
    __device__ uint64_t get_total_deciding_time();
    __device__ uint64_t get_total_backtracking_time();
    __device__ uint64_t get_total_conflict_analyzing_time();

    __device__ int get_sum_of_decision_before_backtracking();
    __device__ int get_sum_of_decision_after_backtracking();
    __device__ uint64_t get_total_backtracked_levels();

    __device__ void signal_start(uint64_t *total_time, int *occurrences);
    __device__ void signal_start(uint64_t *total_time,
                                 int *occurrences, uint64_t time);
    __device__ void signal_stop(uint64_t *total_time);
    __device__ void signal_stop(uint64_t *total_time, uint64_t time);

private:
    int n_blocks;
    int n_threads;

#ifdef ENABLE_STATISTICS
    // Time statistics
    uint64_t *total_solving_time_dev;
    int *n_jobs_run_dev;

    // Partial time
    uint64_t *total_time_analyzing_conflict_dev;
    int *n_conflict_analysis_made_dev;

    uint64_t *total_time_deciding_dev;
    int *n_decisions_made_dev;

    uint64_t *total_time_backtracking_dev;
    int *n_backtracks_made_dev;

    uint64_t *total_time_on_gpu_preprocessing_dev;
    int *n_preprocesses_made_dev;

    uint64_t *total_time_resetting_structures_dev;
    int *n_resets_done;

    uint64_t *total_time_creating_structures_dev;
    int *n_create_structures_done;

    uint64_t *total_time_on_next_job_dev;
    int *n_next_job_done;

    uint64_t *total_time_on_add_jobs_to_assumptions_dev;
    int *n_add_jobs_to_assumptions_done;

    uint64_t *total_time_processing_results_dev;
    int *n_process_results_done;

    // Decisions
    int *sum_of_decision_before_backtracking_dev;
    int *sum_of_decision_after_backtracking_dev;
    uint64_t *total_backtracked_levels_dev;
    int *backtracked_levels_instances_count_dev;
    int *last_backtracked_start_level_dev;

    // Statistics from checking correction
    unsigned *all_threads_total_completed_jobs_dev;

    // Statistics for pre-processing
    uint64_t *total_time_on_gpu_preprocessing_handling_assumption_dev;
    int *n_preprocesses_handling_assumption_made_dev;

    uint64_t *total_time_on_gpu_preprocessing_adding_to_graph_dev;
    int *n_preprocesses_adding_to_graph_made_dev;

    uint64_t *total_time_on_gpu_preprocessing_handling_vars_dev;
    int *n_preprocesses_handling_vars_made_dev;
#endif // ENABLE_STATISTICS

    __device__ int get_index();

    __host__ void clear_stats(int n_blocks, int n_threads);
    __host__ void alloc_stats(int n_blocks, int n_threads);

    template<class T>
    __host__ std::vector<T> copy_to_host(bool block_thread_vector, T *element_dev);

    __host__ void print_stat_average(uint64_t *block_thread_stats_dev, int *count_dev, bool use_offset = true);
    __host__ void print_stat_total(uint64_t *block_thread_stats_dev, int *count_dev);

    template<class T>
    __host__ void print_stat_sum(T *block_thread_stats_dev);
};

template<class T>
__host__ void clear_stat(T *&element, bool block_thread_vector, int n_blocks, int n_threads);
template<class T>
__host__ void alloc_stat(T *&element, bool block_thread_vector, int n_blocks, int n_threads);


#endif /* __RUNTIMESTATISTICS_CUH__ */
