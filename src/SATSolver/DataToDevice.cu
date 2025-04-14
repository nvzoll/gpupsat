#include "DataToDevice.cuh"

/*
DataToDevice::DataToDevice(FormulaData data, int n_jobs, int n_blocks,
            int n_threads, int max_implication_per_var, GPUVec<Var> dead_vars)
{
}
*/
DataToDevice::DataToDevice(CUDAClauseVec const& clauses_database,
                           GPUVec<Var> const& dead_vars,
                           RuntimeStatistics const& st,
                           numbers const& n,
                           atomics const& counters)
    : clauses_db(clauses_database)
    , statistics(st)
    , dead_vars(dead_vars)
    , results(n.vars, true)
    , queue(n.jobs, counters.next_job)
    , nodes_repository(MAX_NUMBER_OF_NODES)
    // , watched_clauses_per_thread(n.blocks*n.threads)

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    , all_assumptions_parallel(n.threads * n.blocks)
    , assumptions_sequential(0)
#endif

    , n_vars { n.vars }
    , n_blocks { n.blocks }
    , n_thread { n.threads }
    , max_implication_per_var { n.max_implication_per_var }

{
    /*
    WatchedClause * all_watched_clauses;

    check(cudaMalloc(&all_watched_clauses, sizeof(WatchedClause)*
            n_blocks*n_threads*n_clauses));

    for (int i = 0; i < n_blocks*n_threads; i++)
    {

        GPUVec <WatchedClause> vec(all_watched_clauses,
                n_clauses, 0);

        watched_clauses_per_thread.add(vec);
        all_watched_clauses+=n_clauses;
    }

    */

}

void DataToDevice::prepare_sequencial()
{

}

void DataToDevice::prepare_parallel(JobChooser& chooser
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    , GPUVec<Lit>& assumptions
#endif
                                   )
{
    if (!chooser.is_evaluated()) {
        chooser.evaluate();
    }

    chooser.getJobs(queue);
    queue.close();

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR

    int largest_job = queue.largest_job_size();

    for (int i = 0; i < n_thread * n_blocks; i++) {
        GPUVec<Lit> *assump = new GPUVec<Lit>(assumptions.size_of() + largest_job);
        all_assumptions_parallel.add(*assump);
    }
#endif


    int init_value = 0;
    check(cudaMalloc(&found_answer, sizeof(unsigned int)), "Allocating data to send to sat_status::SAT Solver.");
    check(cudaMemcpy(found_answer, &init_value, sizeof(unsigned int), cudaMemcpyHostToDevice),
          "Copying data to send to sat_status::SAT Solver.");

}

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
__host__ __device__ GPUVec<GPUVec<Lit>> DataToDevice::get_all_assumptions_parallel()
{
    return all_assumptions_parallel;
}
__device__ GPUVec<Lit> *get_assumptions_sequential()
{
    return assumptions_sequential;
}
#endif

__host__ __device__ JobsQueue& DataToDevice::get_jobs_queue()
{
    return queue;
}

__host__ __device__ JobsQueue *DataToDevice::get_jobs_queue_ptr()
{
    return &queue;
}

__host__ __device__ CUDAClauseVec const& DataToDevice::get_clauses_db()
{
    return clauses_db;
}
__host__ __device__ int DataToDevice::get_n_vars()
{
    return n_vars;
}

__host__ __device__ int DataToDevice::get_max_implication_per_var()
{
    return max_implication_per_var;
}

__host__ __device__ GPUVec<Var> *DataToDevice::get_dead_vars_ptr()
{
    return &dead_vars;
}

__host__ __device__ GPUVec<Var> DataToDevice::get_dead_vars()
{
    return dead_vars;
}

__host__ __device__ RuntimeStatistics& DataToDevice::get_statistics()
{
    return statistics;
}

__host__ __device__ RuntimeStatistics *DataToDevice::get_statistics_ptr()
{
    return &statistics;
}

__host__ __device__ unsigned int *DataToDevice::get_found_answer_ptr()
{
    return found_answer;
}

__host__ __device__ Results DataToDevice::get_results()
{
    return results;
}

__host__ __device__ Results *DataToDevice::get_results_ptr()
{
    return &results;
}

__host__ __device__ NodesRepository<GPULinkedList<WatchedClause *>::Node> *
DataToDevice::get_nodes_repository_ptr()
{
    return &nodes_repository;
}

__host__ __device__ const CUDAClauseVec *DataToDevice::get_clauses_db_ptr()
{
    return &clauses_db;
}

/*
__device__ GPUVec <WatchedClause> DataToDevice::get_watched_clauses(int thread_block_index)
{
    return watched_clauses_per_thread.get(thread_block_index);
}
*/
