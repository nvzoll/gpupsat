#include <stdlib.h>
#include <stdio.h>
#include "Parallelizer.cuh"

__device__ __forceinline__
int get_thread_id() { return threadIdx.x + blockIdx.x * blockDim.x; }

struct KernelContext {
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> *assumptions;
#else
    GPUStaticVec<Lit> the_assumptions;
    GPUStaticVec<Lit> *assumptions;
#endif

    RuntimeStatistics *st;
    JobsQueue *queue;
    SATSolver solver;
    Job job;

    sat_status status = sat_status::UNSAT;

    int solved_jobs;
    // GPUVec <WatchedClause> *watched_clauses;

    unsigned *answer_ptr;

    __device__
    KernelContext(DataToDevice *data)
        : st { data->get_statistics_ptr() }
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
        , assumptions { data->get_all_assumptions_parallel().get_ptr(index) };
#else
        , the_assumptions()
        , assumptions { &the_assumptions }
#endif
        , queue { data->get_jobs_queue_ptr() }
        , solver(data->get_clauses_db_ptr(),
                 data->get_n_vars(),
                 data->get_max_implication_per_var(),
                 data->get_dead_vars_ptr(),
                 st,
                 data->get_nodes_repository_ptr()
                 /*, watched_clauses*/)
        , solved_jobs { 0 }
        , answer_ptr { data->get_found_answer_ptr() }
        // , watched_clauses { data.get_watched_clauses(index) }
    {
        next_job_available();
    }

    __device__
    ~KernelContext() { }

    __device__
    void solve()
    {
        if (job.n_literals == 0
            && !next_job_available()) {
            return;
        }

        status = solver.solve(assumptions);
    }

    __device__
    bool finished()
    {
        if (status == sat_status::SAT
            || status == sat_status::UNDEF) {
            return true;
        }

        reset_solver();
        remove_last_assumptions();

        return !next_job_available();
    }

    __device__
    bool next_job_available()
    {
        st->signal_next_job_start();
        job = queue->next_job();
        st->signal_next_job_stop();

        return (job.n_literals != 0);
    }

    __device__
    void reset_solver()
    {
        st->signal_reset_structures_start();
        solver.reset();
        st->signal_reset_structures_stop();
    }

    __device__
    void remove_last_assumptions()
    {
        st->signal_add_jobs_to_assumptions_start();
        assumptions->remove_n_last(job.n_literals);
        st->signal_add_jobs_to_assumptions_stop();
    }

    __device__
    void add_assumptions()
    {
        st->signal_add_jobs_to_assumptions_start();

        for (int i = 0; i < job.n_literals; i++) {
            assumptions->add(job.literals[i]);
        }

        st->signal_add_jobs_to_assumptions_stop();
    }
};

__device__ __forceinline__
KernelContext *get_ctx(KernelContextStorage *storage)
{
    KernelContext **row = (KernelContext **)((char *)storage->data + blockIdx.x * storage->pitch);
    return row[threadIdx.x];
}

__device__ __forceinline__
void save_ctx(KernelContext *ctx, KernelContextStorage *storage)
{
    KernelContext **row = (KernelContext **)((char *)storage->data + blockIdx.x * storage->pitch);
    row[threadIdx.x] = ctx;
}

__global__ void parallel_kernel_init(DataToDevice *data, KernelContextStorage storage)
{
    int index = get_thread_id();

    RuntimeStatistics *st = data[index].get_statistics_ptr();

    st->signal_create_structures_start();
        KernelContext *ctx = new KernelContext(&data[index]);
    st->signal_create_structures_stop();

    save_ctx(ctx, &storage);

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("sizeof(KernelContext)=0x%x\n", sizeof(KernelContext));
    }

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Block: %d, thread: %d, index: %d\n", blockIdx.x, threadIdx.x, index );
        printf("Largest job is : %d\n", ctx->queue->largest_job_size());
        printf("Number of literals: %d\n", ctx->job.n_literals);
    }
}

__global__ void parallel_kernel_retrieve_results(DataToDevice data, KernelContextStorage storage)
{
    KernelContext *ctx = get_ctx(&storage);
    RuntimeStatistics *st = ctx->st;

    st->signal_process_results_start();

    if (ctx->status == sat_status::SAT) {
        size_t results_size = ctx->solver.get_results_size();
        Lit *content = new Lit[results_size];
        ctx->solver.get_results(content);

        Results *results = data.get_results_ptr();
        results->set_satisfiable_results(content, results_size);

        delete[] content;
    }

    if (ctx->status == sat_status::UNDEF) {
        Results results = data.get_results();
        int first = atomicInc(data.get_found_answer_ptr(), 1000);
        printf("Block %d, thread %d had a problem\n", blockIdx.x, threadIdx.x);
        results.set_undef();
    }

    st->signal_process_results_stop();
}

__global__ void parallel_kernel(KernelContextStorage storage, int *state)
{
    int index = get_thread_id();

    KernelContext *ctx = get_ctx(&storage);
    RuntimeStatistics *st = ctx->st;

    ctx->solved_jobs++;

    st->signal_job_start();

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Thread of index %d is running job: ", index);
        print_job(ctx->job);
        printf("Job has %d literals", ctx->job.n_literals);
        printf("\n");
    }

    ctx->add_assumptions();

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("The jobs as assumptions are: ");
        for (int i = 0; i < ctx->assumptions->size_of(); i++) {
            print_lit(ctx->assumptions->get(i));
            printf(" ");
        }
        printf("\n");
    }

    ctx->solve();

    st->signal_job_stop();

    if (!ctx->finished()) {
        return;
    }

    // TODO replace 1000 for something else.
    int first = atomicInc(ctx->answer_ptr, 1000);
    if (first == 0) {
        *state = index;

        st->add_completed_jobs(ctx->solved_jobs);

#ifdef DEBUG
        printf("End of loop.\n");
#endif
    }
}

__global__ void run_sequential(DataToDevice data, int *state)
{
    RuntimeStatistics *st = data.get_statistics_ptr();
    st->signal_create_structures_start();

    const CUDAClauseVec *clauses_db = data.get_clauses_db_ptr();
    int n_vars = data.get_n_vars();

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> assumptions = data.get_assumptions_sequential();
#else
    //GPUStaticVec<Lit> assumptions = data.get_assumptions_sequential();
#endif

    Results  results = data.get_results();

    //GPUVec <WatchedClause> watched_clauses = data.get_watched_clauses(0);

    SATSolver *solver = new SATSolver(clauses_db,
                                      n_vars,
                                      data.get_max_implication_per_var(),
                                      data.get_dead_vars_ptr(),
                                      st,
                                      data.get_nodes_repository_ptr()
                                      //, watched_clauses
                                     );

    st->signal_create_structures_stop();
    st->signal_job_start();

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    sat_status status = solver->solve(&assumptions);
#else
    sat_status status = solver->solve();
#endif

    st->signal_job_stop();

    st->signal_process_results_start();

    if (status == sat_status::SAT) {
        int results_size = solver->get_results_size();
        Lit *the_results = new Lit[results_size];
        solver->get_results(the_results);

        results.set_satisfiable_results(the_results, results_size);

        delete[] the_results;
    }

    if (status == sat_status::UNSAT) {
        // do nothing!
    }

    if (status == sat_status::UNDEF) {
        results.set_undef();
    }

    st->signal_process_results_stop();

    *state = get_thread_id();

    delete solver;
}

