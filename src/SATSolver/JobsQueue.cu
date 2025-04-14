#include "JobsQueue.cuh"

JobsQueue::JobsQueue(size_t capacity, unsigned *atomic_counter)
    : jobs(capacity)
    , next_job_index { atomic_counter }
{

}

__device__ Job JobsQueue::next_job()
{
    if (!closed) {
        printf("(UNSAFE) Should not get objects before closing!\n");
    }

    unsigned index = atomicInc(next_job_index, LARGE_NUMBER);

#ifdef DEBUG
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Chose job of index %d of %d jobs\n", index, jobs.size_of());
    }
#endif

    if (index >= this->jobs.size_of()) {
        Job j;
        j.n_literals = 0;
        return j;
    }
    else {
        return jobs.get(index);
    }
}

__host__ __device__ void JobsQueue::close()
{
    closed = true;
}

__host__ void JobsQueue::add(Job& job)
{

    if (closed) {
        printf("Cannot add after closed.\n");
        return;
    }

    jobs.add(job);


    if (job.n_literals > size_of_largest_job) {
        size_of_largest_job = job.n_literals;
    }
}

__host__ __device__ void JobsQueue::print_jobs()
{
    printf("There are %zu jobs\n", n_jobs());

    for (size_t i = 0; i < n_jobs(); i++) {
        Job j = jobs.get(i);
        print_job(j);
    }
}

__host__ Job mkJob_dev(Lit *literals, size_t n_literals)
{
    Lit *dev_lits;

    check(cudaMalloc(&dev_lits, n_literals * sizeof(Lit)), "Allocating job on dev");

    check(cudaMemcpy(dev_lits, literals, n_literals * sizeof(Lit),
                     cudaMemcpyHostToDevice), "Copying job to dev");

    Job job = { dev_lits, n_literals };

    return job;

}

__host__ __device__ void print_job(Job& job)
{
#ifdef __CUDA_ARCH__
    printf("Job: ");
    for (size_t i = 0; i < job.n_literals; i++) {
        print_lit(job.literals[i]);
        printf(" ");
    }

    printf("\n");
#else
    Lit *literals_host = new Lit[job.n_literals];

    cudaMemcpy(literals_host, job.literals,
               sizeof(Lit)*job.n_literals, cudaMemcpyDeviceToHost);

    printf("Job: ");
    for (size_t i = 0; i < job.n_literals; i++) {
        print_lit(literals_host[i]);
        printf(" ");
    }
    printf("\n");

    delete[] literals_host;
#endif
}
