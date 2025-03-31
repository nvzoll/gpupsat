#include "Parallelizer.cuh"
#include <stdio.h>
#include <stdlib.h>

// Dummy kernel for testing launch stability
__global__ void dummy_kernel() { }

__device__ __forceinline__ int get_thread_id() { return threadIdx.x + blockIdx.x * blockDim.x; }

struct KernelContext {
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit>* assumptions;
#else
    GPUStaticVec<Lit> the_assumptions;
    GPUStaticVec<Lit>* assumptions;
#endif

    RuntimeStatistics* st;
    JobsQueue* queue;
    SATSolver solver;
    Job job; // Holds offset and n_literals

    // Added pointer to the base of the shared literal buffer
    Lit* literal_buffer_base_ptr;

    sat_status status = sat_status::UNSAT;

    int solved_jobs;
    // GPUVec <WatchedClause> *watched_clauses;

    unsigned* answer_ptr;
    Results* results_ptr; // Added pointer to Results object

    __device__
    // Constructor updated to accept KernelDataPod* and index, and initialize new pointer
    KernelContext(KernelDataPod* pod, int index) // Added index parameter
        : st { pod->statistics_ptr } // Access via pod member
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
                                     // TODO: Need to handle assumptions pointer correctly via pod
        , assumptions { pod->all_assumptions_parallel_ptr ? pod->all_assumptions_parallel_ptr->get_ptr(index) : nullptr } // Example, needs verification
#else
        , the_assumptions()
        , assumptions { &the_assumptions } // Keep static assumptions as is for now
#endif
        // Update other initializers to use pod members
        , queue { pod->queue_ptr }
        ,
        // Pass raw pointer and size for dead_vars to SATSolver constructor
        solver(pod->clauses_db_ptr, pod->n_vars, pod->max_implication_per_var,
            pod->dead_vars_elements_ptr, pod->dead_vars_size, // Use new pod members
            st, pod->nodes_repository_ptr
            /*, watched_clauses*/)
        , // Note: 'st' is already initialized above
        // Initialize the literal buffer base pointer from the pod
        literal_buffer_base_ptr { pod->literal_buffer_base_ptr }
        , results_ptr { pod->results_ptr }
        , // Initialize results pointer
        solved_jobs { 0 }
        , answer_ptr { pod->found_answer_ptr } // Use pod member
    // , watched_clauses { ??? } // Need to decide how to handle this if needed
    {
        // Get the first job
        next_job_available();
    }

    __device__ ~KernelContext() { }

    __device__ void solve()
    {
        // Check if the current job is valid before solving
        if (job.n_literals == 0) // If no valid job was fetched, don't solve
        {
            // Try fetching a new job if the first one was invalid/end signal
            if (!next_job_available())
                return; // No more jobs, exit
        }

        // Add assumptions for the current valid job
        add_assumptions();

        // Solve with the added assumptions
        status = solver.solve(assumptions);
    }

    __device__ bool finished()
    {
        if (status == sat_status::SAT || status == sat_status::UNDEF) {
            return true; // Found SAT or encountered an error
        }

        // If UNSAT, remove assumptions from the completed job
        remove_last_assumptions();
        // Reset solver state for the next job
        reset_solver();

        // Try to get the next job
        return !next_job_available(); // Return true if no more jobs are available
    }

    __device__ bool next_job_available()
    {
        st->signal_next_job_start();
        job = queue->next_job(); // Fetch the next job (contains offset and size)
        st->signal_next_job_stop();

        return (job.n_literals != 0); // Return true if a valid job was fetched
    }

    __device__ void reset_solver()
    {
        st->signal_reset_structures_start();
        solver.reset();
        st->signal_reset_structures_stop();
    }

    __device__ void remove_last_assumptions()
    {
        // Only remove if the last job was valid
        if (job.n_literals > 0) {
            st->signal_add_jobs_to_assumptions_start();
            assumptions->remove_n_last(job.n_literals);
            st->signal_add_jobs_to_assumptions_stop();
        }
    }

    // Modified to use the literal buffer base pointer and the job's offset
    __device__ void add_assumptions()
    {
        // Only add if the job is valid and the base pointer is set
        if (job.n_literals > 0 && literal_buffer_base_ptr != nullptr) {
            st->signal_add_jobs_to_assumptions_start();

            // Calculate the pointer to the literals for the current job
            Lit* job_literals = literal_buffer_base_ptr + job.literal_offset;

            for (size_t i = 0; i < job.n_literals; i++) // Use size_t for loop
            {
                // Add the literal from the calculated position in the shared buffer
                assumptions->add(job_literals[i]);
            }

            st->signal_add_jobs_to_assumptions_stop();
        } else if (literal_buffer_base_ptr == nullptr && job.n_literals > 0) {
            // Optional: Add a warning if the pointer is null but we have a job
            // printf("Warning: literal_buffer_base_ptr is null in add_assumptions for job with offset %zu\n", job.literal_offset);
        }
    }
};

__device__ __forceinline__ KernelContext* get_ctx(KernelContextStorage* storage)
{
    KernelContext** row = (KernelContext**)((char*)storage->data + blockIdx.x * storage->pitch);
    return row[threadIdx.x];
}

__device__ __forceinline__ void save_ctx(KernelContext* ctx, KernelContextStorage* storage)
{
    KernelContext** row = (KernelContext**)((char*)storage->data + blockIdx.x * storage->pitch);
    row[threadIdx.x] = ctx;
}

// Kernel to initialize the context for each thread
__global__ void parallel_kernel_init(KernelDataPod* pod_data, KernelContextStorage storage)
{
    int index = get_thread_id();

    // Ensure index is within bounds of the pod_data buffer
    // This assumes pod_data has size n_blocks * n_threads
    // No explicit check here, relying on correct launch configuration

    // Access the specific pod for this thread
    KernelDataPod* thread_pod = &pod_data[index];

    // RuntimeStatistics *st = thread_pod->statistics_ptr;

    // st->signal_create_structures_start();
    // Allocate and construct KernelContext using 'new' in device code
    KernelContext* ctx = new KernelContext(thread_pod, index); // Pass the specific pod pointer
    // st->signal_create_structures_stop();

    // Save the pointer to the allocated context in the storage array
    save_ctx(ctx, &storage);

    // Debug printing (optional)
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("sizeof(KernelContext)=%llu\n", sizeof(KernelContext)); // Use %zu for size_t
        printf("Block: %d, thread: %d, index: %d\n", blockIdx.x, threadIdx.x, index);
        // Check if queue pointer is valid before dereferencing
        if (ctx->queue) {
            printf("Largest job is : %llu\n", ctx->queue->largest_job_size()); // Use %zu
        } else {
            printf("Queue pointer is null\n");
        }
        printf("Initial job n_literals: %llu\n", ctx->job.n_literals); // Use %zu
    }
}

// Removed parallel_kernel_retrieve_results kernel definition

// Main parallel execution kernel
__global__ void parallel_kernel(KernelContextStorage storage, int* state)
{
    int index = get_thread_id();

    // Retrieve this thread's context
    KernelContext* ctx = get_ctx(&storage);
    if (!ctx)
        return; // Safety check

    RuntimeStatistics* st = ctx->st;
    if (!st)
        return; // Safety check

    // Check if a solution has already been found by another thread
    if (*state != INT_MAX) {
        return; // Exit early if solution found
    }

    // Check if the context has a valid job to start with
    if (ctx->job.n_literals == 0) {
        // If no valid job initially, and no more jobs available, this thread is done.
        if (!ctx->next_job_available()) {
            return;
        }
    }

    // Main loop: process jobs until a result is found or no jobs remain
    while (true) {
        // Check again if a solution was found while this thread was working
        if (*state != INT_MAX) {
            return;
        }

        st->signal_job_start();

        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("Thread %d running job (offset %zu, n_lits %zu): ", index, ctx->job.literal_offset, ctx->job.n_literals);
            // Pass the base pointer to print_job
            if (ctx->queue) { // Check queue pointer
                print_job(ctx->job, ctx->literal_buffer_base_ptr); // Pass base pointer
            } else {
                printf("[Queue ptr null]\n");
            }
        }

        // Solve the current job (includes adding assumptions)
        ctx->solve();

        st->signal_job_stop();

        // Check if this job finished the search (SAT, UNDEF, or no more jobs)
        if (ctx->finished()) {
            // If SAT or UNDEF found by this thread
            if (ctx->status == sat_status::SAT || ctx->status == sat_status::UNDEF) {
                // Try to claim the global state using atomic compare-and-swap
                int old_state = atomicCAS(state, INT_MAX, index);
                if (old_state == INT_MAX) {
                    // This thread is the first to find a result
                    // Set the result directly here using the context's solver state
                    if (ctx->status == sat_status::SAT) {
                        // Access the Results object via the pointer stored in the context
                        Results* results_obj = ctx->results_ptr;
                        if (results_obj) {
                            size_t results_size = ctx->solver.get_results_size();
                            // Allocate temporary buffer on device stack or use pre-allocated buffer if possible
                            // WARNING: 'new' in device code can be problematic (performance, fragmentation).
                            // Consider using stack allocation (if size is reasonably small and known)
                            // or a pre-allocated shared/global buffer.
                            Lit* content = new Lit[results_size];
                            ctx->solver.get_results(content);
                            results_obj->set_satisfiable_results(content, results_size);
                            delete[] content; // Free temporary buffer
                        }
                    } else { // UNDEF
                        // Access the Results object via the pointer stored in the context
                        Results* results_obj = ctx->results_ptr;
                        if (results_obj) {
                            results_obj->set_undef();
                        }
                    }

                    st->add_completed_jobs(ctx->solved_jobs + 1); // Add jobs completed by this thread
                    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
                        printf("Thread %d finished with status %d.\n", index, ctx->status);
                    }
                }
                // Whether this thread won or not, it's done.
                return;
            } else {
                // If finished() returned true but status is UNSAT, it means no more jobs.
                // This thread is done.
                return;
            }
        }
        // If finished() returned false, it means the job was UNSAT,
        // assumptions were removed, solver reset, and a new job was fetched (or tried).
        // Continue the loop with the next job.
        ctx->solved_jobs++; // Increment solved count for the job just completed (UNSAT)
    } // End while(true) loop
}

// Sequential execution kernel (remains largely unchanged for now)
__global__ void run_sequential(DataToDevice data, int* state)
{
    RuntimeStatistics* st = data.get_statistics_ptr();
    st->signal_create_structures_start();

    const CUDAClauseVec* clauses_db = data.get_clauses_db_ptr();
    int n_vars = data.get_n_vars();

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    // Need to get the correct assumptions vector for sequential
    // GPUVec<Lit>* assumptions = data.get_assumptions_sequential(); // Assuming this getter exists and is correct
#endif

    Results* results = data.get_results_ptr(); // Get pointer

    // GPUVec <WatchedClause> watched_clauses = data.get_watched_clauses(0);

    // Get the GPUVec pointer, then extract raw pointer and size
    GPUVec<Var>* dead_vars_gpuvec_ptr = data.get_dead_vars_ptr();
    const Var* dead_vars_elements = dead_vars_gpuvec_ptr ? dead_vars_gpuvec_ptr->data() : nullptr;
    size_t dead_vars_count = dead_vars_gpuvec_ptr ? dead_vars_gpuvec_ptr->size_of() : 0;

    SATSolver* solver = new SATSolver(clauses_db,
        n_vars,
        data.get_max_implication_per_var(),
        dead_vars_elements, // Pass raw pointer
        dead_vars_count, // Pass size
        st,
        data.get_nodes_repository_ptr()
        //, watched_clauses
    );

    st->signal_create_structures_stop();
    st->signal_job_start();

    sat_status status;
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    // Pass pointer if solve expects it
    // status = solver->solve(assumptions);
    status = solver->solve(); // Assuming solve() without args if no assumptions needed
#else
    status = solver->solve();
#endif

    st->signal_job_stop();

    st->signal_process_results_start();

    if (results) { // Check if results pointer is valid
        if (status == sat_status::SAT) {
            int results_size = solver->get_results_size();
            Lit* the_results = new Lit[results_size]; // Allocate on device? Host? Needs clarification. Assuming host for now.
            solver->get_results(the_results); // Assuming get_results copies to host pointer

            results->set_satisfiable_results(the_results, results_size);

            delete[] the_results; // Free host memory
        } else if (status == sat_status::UNSAT) {
            // Do nothing - UNSAT is the implicit status if not SAT/UNDEF
        } else // UNDEF
        {
            results->set_undef();
        }
    }

    st->signal_process_results_stop();

    *state = get_thread_id(); // Mark sequential run as complete (index 0)

    delete solver; // Free solver allocated with new
}
