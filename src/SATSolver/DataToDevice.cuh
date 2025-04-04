#ifndef __DATATODEVICE_CUH__
#define __DATATODEVICE_CUH__

#include "JobsQueue.cuh" // Moved earlier
#include "Utils/GPUVec.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "Results.cuh"
#include "JobsManager/JobChooser.cuh"
#include "BCPStrategy/WatchedClausesList.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"
#include "Utils/NodesRepository.cuh"
#include "FileManager/FormulaData.cuh"

class DataToDevice
{
public:
    struct numbers
    {
        int vars;
        int clauses;
        int jobs;
        int blocks;
        int threads;
        int max_implication_per_var;
    };

    struct atomics
    {
        unsigned *next_job;
        unsigned *completed_jobs;
    };

    // Constructor updated to accept pre-calculated capacities for JobsQueue and RuntimeStatistics pointer
    DataToDevice(CUDAClauseVec const &clauses_database,
                 GPUVec<Var> const &dead_vars,
                 RuntimeStatistics *statistics_ptr, // Changed to pointer
                 numbers const &n,                  // Keep for other numbers like n_vars, blocks, threads
                 atomics const &counters,
                 size_t job_capacity,      // For JobsQueue jobs vector
                 size_t literal_capacity); // For JobsQueue literal buffer

    //    DataToDevice(FormulaData data, int n_jobs, int n_blocks,
    //            int n_threads, int max_implication_per_var, GPUVec<Var> dead_vars); // Old comment

    void prepare_sequencial();
    void prepare_parallel(JobChooser &chooser
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
                          ,
                          GPUVec<Lit> &assumptions
#endif
    );

    __host__ void cleanup(); // Added cleanup function declaration

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ GPUVec<GPUVec<Lit>> *get_all_assumptions_parallel();
    __device__ GPUVec<Lit> *get_assumptions_sequential();
#endif

    __host__ __device__ JobsQueue &get_jobs_queue();
    __host__ __device__ JobsQueue *get_jobs_queue_ptr();
    __host__ __device__ CUDAClauseVec const &get_clauses_db();
    __host__ __device__ const CUDAClauseVec *get_clauses_db_ptr();
    __host__ __device__ int get_n_vars();
    __host__ __device__ int get_max_implication_per_var();
    __host__ __device__ GPUVec<Var> *get_dead_vars_ptr();
    __host__ __device__ GPUVec<Var> get_dead_vars();

    // Removed get_statistics() as we only store a pointer now
    __host__ __device__ RuntimeStatistics *get_statistics_ptr();

    __host__ __device__ unsigned int *get_found_answer_ptr();
    __host__ __device__ Results get_results();
    __host__ __device__ Results *get_results_ptr();

    __host__ __device__ watched_clause_node_t *get_nodes_repository_ptr();
    //__device__ GPUVec <WatchedClause> get_watched_clauses(int thread_block_index);

private:
    const CUDAClauseVec clauses_db;
    const int n_vars;
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<GPUVec<Lit>> all_assumptions_parallel;
    GPUVec<Lit> assumptions_sequential;
#endif
    JobsQueue queue;
    unsigned int *found_answer;

    RuntimeStatistics *statistics_ptr; // Changed to pointer

    Results results;
    int max_implication_per_var;
    GPUVec<Var> dead_vars;
    int n_thread;
    int n_blocks;

    // GPUVec< GPUVec < WatchedClause > > watched_clauses_per_thread;
    NodesRepository<GPULinkedList<WatchedClause *>::Node> nodes_repository;
};

#endif /* __DATATODEVICE_CUH__ */
