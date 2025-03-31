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
struct Job
{
    size_t literal_offset; // Offset into the shared literal buffer
    size_t n_literals;
};

// __host__ Job mkJob_dev(Lit *literals, size_t n_literals); // Becomes internal detail
__host__ __device__ void print_job(Job &job, Lit *literal_buffer_base_ptr); // Needs base pointer

class JobsQueue
{
private:
    GPUVec<Job> jobs;
    GPUVec<Lit> all_literals_buffer; // Single buffer for all literals
    size_t current_literal_offset;   // Tracks next available offset in buffer
    unsigned *next_job_index;
    bool closed = false;
    size_t size_of_largest_job = 0;

public:
    // Constructor now needs literal capacity as well
    JobsQueue(size_t job_capacity, size_t literal_capacity, unsigned *atomic_counter);

    __device__ Job next_job();
    __host__ __device__ void close();
    // __host__ void add(Job &job); // Old signature
    __host__ void add(Lit *host_literals, size_t n_literals); // New signature
    __host__ __device__ void print_jobs();
    __host__ void free_job_literals(); // Now simplifies to freeing the single buffer

    // New methods for buffer management
    __host__ void reserve_literal_buffer(size_t total_literal_count);
    __host__ __device__ Lit *get_literal_buffer_ptr();

    __host__ __device__ size_t largest_job_size() const { return size_of_largest_job; }
    __host__ __device__ size_t n_jobs() const { return jobs.size_of(); }
};
#endif /* __JOBSQUEUE_CUH__ */
