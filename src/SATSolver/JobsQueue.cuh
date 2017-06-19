#ifndef __JOBSQUEUE_CUH__
#define __JOBSQUEUE_CUH__

#include "Utils/GPUVec.cuh"
#include "Configs.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"
#include <climits>

/**
 * A struct that represents a job and must always be on the device.
 * IMPORTANT: The Lit pointer must point to a vector on the device!
 *
 * n_literals = 0 -> end job: signal that there are not any more jobs.
 *
 */
struct Job {
    Lit *literals;
    size_t n_literals;
};

__host__ Job mkJob_dev(Lit *literals, size_t n_literals);
__host__ __device__ void print_job(Job& job);

class JobsQueue
{
private:
    GPUVec<Job> jobs;
    unsigned *next_job_index;
    bool closed = false;
    size_t size_of_largest_job = 0;

public:
    JobsQueue(size_t capacity, unsigned *atomic_counter);

    __device__ Job next_job();
    __host__ __device__ void close();
    __host__ void add(Job& job);
    __host__ __device__ void print_jobs();

    __host__ __device__ size_t largest_job_size() const { return size_of_largest_job; }
    __host__ __device__ size_t n_jobs() const { return jobs.size_of(); }
};
#endif /* __JOBSQUEUE_CUH__ */

