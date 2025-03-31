#include "DataToDevice.cuh"
#include <cuda_runtime.h> // For cudaMalloc/cudaMemcpy/cudaFree
#include <stdio.h>        // For printf

/*
DataToDevice::DataToDevice(FormulaData data, int n_jobs, int n_blocks,
            int n_threads, int max_implication_per_var, GPUVec<Var> dead_vars)
{
}
*/

// Constructor definition updated to match header (accepts RuntimeStatistics*) and pass capacities to JobsQueue
DataToDevice::DataToDevice(CUDAClauseVec const &clauses_database,
                           GPUVec<Var> const &dead_vars_ref, // Renamed to avoid shadowing member
                           RuntimeStatistics *st_ptr,        // Changed to pointer
                           numbers const &n,
                           atomics const &counters,
                           size_t job_capacity,     // For JobsQueue jobs vector
                           size_t literal_capacity) // For JobsQueue literal buffer
    : clauses_db(clauses_database),
      statistics_ptr(st_ptr),   // Store the pointer directly
      dead_vars(dead_vars_ref), // Copy construct dead_vars member
      results(n.vars, true),
      // Pass capacities and counter to JobsQueue constructor
      queue(job_capacity, literal_capacity, counters.next_job),
      nodes_repository(MAX_NUMBER_OF_NODES)
// , watched_clauses_per_thread(n.blocks*n.threads)

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
      ,
      // Initialize assumption vectors (capacity might need adjustment based on usage)
      all_assumptions_parallel(n.threads * n.blocks),
      assumptions_sequential(0)
#endif

      ,
      n_vars{n.vars},
      n_blocks{n.blocks},
      n_thread{n.threads},
      max_implication_per_var{n.max_implication_per_var},
      found_answer{nullptr} // Initialize found_answer pointer

{
    // Constructor body - Allocate found_answer here? Or in prepare_parallel?
    // Let's keep it in prepare_parallel for now as it seems related to parallel run setup.
}

void DataToDevice::prepare_sequencial()
{
    // Allocate found_answer if needed for sequential run?
    // For now, assume sequential doesn't use it or handles it separately.
}

// Updated prepare_parallel: Removed buffer reservation (done in constructor)
void DataToDevice::prepare_parallel(JobChooser &chooser
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
                                    ,
                                    GPUVec<Lit> &assumptions // This parameter seems unused now?
#endif
)
{
    // Ensure chooser is evaluated (might have been done earlier in main)
    if (!chooser.is_evaluated())
    {
        chooser.evaluate();
    }

    // Populate the queue (which now copies literals to the pre-allocated buffer)
    chooser.getJobs(queue);
    queue.close(); // Mark queue as ready for consumption

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    // This logic might need review. Does it still allocate GPUVec<Lit> per thread?
    // If so, it might need similar refactoring or careful cleanup.
    // For now, leaving it as is but noting potential issue.
    int largest_job = queue.largest_job_size();
    for (int i = 0; i < n_thread * n_blocks; i++)
    {
        // WARNING: This 'new GPUVec<Lit>' allocates on HOST heap, not device,
        // and its internal 'elements' likely allocates on DEVICE via cudaMalloc.
        // This needs careful cleanup in DataToDevice::cleanup.
        GPUVec<Lit> *assump = new GPUVec<Lit>(assumptions.size_of() + largest_job); // assumptions param seems unused?
        all_assumptions_parallel.add(*assump);                                      // Adds the GPUVec object itself, not its elements
        // Potential memory leak if 'assump' pointer is not stored/deleted later.
        // Also, GPUVec::add likely copies the GPUVec object, potentially shallow-copying 'elements'.
        // This whole section needs careful review based on GPUVec implementation details.
    }
#endif

    // Allocate the shared 'found_answer' flag on the device
    unsigned int init_value = 0; // Use unsigned int to match pointer type
    check(cudaMalloc(&found_answer, sizeof(unsigned int)), "Allocating found_answer flag.");
    check(cudaMemcpy(found_answer, &init_value, sizeof(unsigned int), cudaMemcpyHostToDevice),
          "Initializing found_answer flag.");
}

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
// These might need adjustment based on how all_assumptions_parallel is managed
__host__ __device__ GPUVec<GPUVec<Lit>> *DataToDevice::get_all_assumptions_parallel()
{
    return &all_assumptions_parallel;
}
__device__ GPUVec<Lit> *DataToDevice::get_assumptions_sequential()
{
    return &assumptions_sequential;
}
#endif

__host__ __device__ JobsQueue &DataToDevice::get_jobs_queue()
{
    return queue;
}

__host__ __device__ JobsQueue *DataToDevice::get_jobs_queue_ptr()
{
    return &queue;
}

__host__ __device__ CUDAClauseVec const &DataToDevice::get_clauses_db()
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

// Removed DataToDevice::get_statistics() implementation as the method was removed from the header.

__host__ __device__ RuntimeStatistics *DataToDevice::get_statistics_ptr()
{
    // Return the stored pointer directly
    return statistics_ptr;
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

// Updated cleanup to explicitly call destroy() on owned GPUVec members
__host__ void DataToDevice::cleanup()
{
    printf("Cleaning up DataToDevice resources...\n");

    // Cleanup JobsQueue (which handles its internal buffers)
    queue.free_job_literals(); // Calls destroy() on queue's GPUVec members

    // Free 'found_answer' pointer allocated in prepare_parallel
    if (found_answer != nullptr)
    {
        check(cudaFree(found_answer), "Freeing found_answer pointer in DataToDevice::cleanup");
        found_answer = nullptr; // Good practice
    }

    // Explicitly destroy other owned GPUVec members
    dead_vars.destroy(); // Was copy-constructed

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    // Need to handle cleanup for all_assumptions_parallel carefully.
    // If it stores GPUVec objects directly, need to iterate and destroy them.
    // If it stores pointers allocated with 'new', need to iterate, destroy pointed-to objects, then delete pointers.
    // The current implementation seems problematic (see notes in prepare_parallel).
    // Assuming for now it stores objects and needs destroy called on each.
    // This requires GPUVec::copyToHost or similar host-side access.
    /*
    GPUVec<GPUVec<Lit>>* host_assumptions_vec = all_assumptions_parallel.copyToHost(); // Hypothetical
    if (host_assumptions_vec) {
        for (size_t i = 0; i < all_assumptions_parallel.size_of(); ++i) {
             host_assumptions_vec[i].destroy(); // Destroy each inner GPUVec<Lit>
        }
        delete[] host_assumptions_vec;
    }
    */
    // Then destroy the outer vector
    all_assumptions_parallel.destroy();
    assumptions_sequential.destroy();
    printf("Cleaned up assumption vectors (potential issues remain).\n");
#endif

    // clauses_db is const& - not owned
    // statistics_ptr is pointer to managed memory - not owned by DataToDevice
    // results is member object - destructor handles cleanup
    // nodes_repository is member object - destructor handles cleanup

    printf("Finished cleaning up DataToDevice resources.\n");
}
