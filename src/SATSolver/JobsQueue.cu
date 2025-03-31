#include "JobsQueue.cuh"
#include <cuda_runtime.h> // Needed for cudaMemcpy
#include <stdio.h> // Needed for printf

// Constructor updated for new members and parameter name
// Takes job_capacity for the jobs vector and literal_capacity for the shared buffer
JobsQueue::JobsQueue(size_t job_capacity, size_t literal_capacity, unsigned* atomic_counter)
    : jobs(job_capacity)
    , // Capacity for Job structs
    all_literals_buffer(literal_capacity)
    , // Initialize literal buffer with fixed capacity
    current_literal_offset { 0 }
    , // Start offset at 0
    next_job_index { atomic_counter }
    , closed { false }
    , size_of_largest_job { 0 }
{
    // Constructor body might be empty if all initialization is done above
}

__device__ Job JobsQueue::next_job()
{
    // This part remains largely the same
    if (!closed) {
        // printf("(UNSAFE) Should not get objects before closing!\n");
    }

    unsigned index = atomicInc(next_job_index, LARGE_NUMBER);

#ifdef DEBUG
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Chose job of index %d of %d jobs\n", index, jobs.size_of());
    }
#endif

    if (index >= this->jobs.size_of()) {
        Job j;
        j.literal_offset = 0; // Or some invalid offset?
        j.n_literals = 0; // Signal end job
        return j;
    } else {
        return jobs.get(index);
    }
}

__host__ __device__ void JobsQueue::close()
{
    closed = true;
}

// Removed reserve_literal_buffer method implementation

// Updated host/device method to get the base pointer using data()
__host__ __device__ Lit* JobsQueue::get_literal_buffer_ptr()
{
    return all_literals_buffer.data(); // Use the new data() method
}

// Modified host method to add a job by copying literals to the shared buffer
__host__ void JobsQueue::add(Lit* host_literals, size_t n_literals)
{
    if (closed) {
        printf("Cannot add after closed.\n");
        return;
    }

    // Removed capacity check - relying on correct pre-calculation passed to constructor

    // Copy literals from host to the device buffer at the current offset
    // Use data() to get the base pointer
    Lit* buffer_target_ptr = all_literals_buffer.data() + current_literal_offset;
    check(cudaMemcpy(buffer_target_ptr, host_literals, n_literals * sizeof(Lit), cudaMemcpyHostToDevice),
        "Copying job literals to shared buffer");

    // Create the Job struct with the offset
    Job job = { current_literal_offset, n_literals };

    // Add the Job struct to the jobs vector
    jobs.add(job);

    // Update the offset for the next job
    current_literal_offset += n_literals;

    // Update largest job size if needed
    if (job.n_literals > size_of_largest_job) {
        size_of_largest_job = job.n_literals;
    }
}

// Modified to require the base pointer
__host__ __device__ void JobsQueue::print_jobs()
{
    printf("There are %zu jobs\n", n_jobs());

#ifndef __CUDA_ARCH__
    // Host-side printing needs the base pointer
    Lit* base_ptr = all_literals_buffer.data(); // Use data()
    if (!base_ptr && n_jobs() > 0) {
        printf("Warning: Cannot print job literals on host, buffer pointer is null.\n");
        for (size_t i = 0; i < n_jobs(); i++) {
            Job j = jobs.get(i);
            printf("Job %zu: offset=%zu, n_literals=%zu\n", i, j.literal_offset, j.n_literals);
        }
        return;
    }
#else
    // Device-side printing needs the base pointer passed somehow.
    // Assuming it's available in the scope calling this.
    Lit* base_ptr = all_literals_buffer.data(); // Placeholder for device context
#endif

    for (size_t i = 0; i < n_jobs(); i++) {
#ifdef __CUDA_ARCH__
        Job j = jobs.get(i);
        print_job(j, base_ptr);
#else
        Job j = jobs.get(i);
        print_job(j, base_ptr);
#endif
    }
}

// Removed __host__ Job mkJob_dev(...) function

// Modified to use offset and base pointer
__host__ __device__ void print_job(Job& job, Lit* literal_buffer_base_ptr)
{
    if (job.n_literals == 0) {
        printf("Job: [End Job Signal]\n");
        return;
    }
    if (!literal_buffer_base_ptr) {
        printf("Job (offset %zu, n_lits %zu): [Error: Base pointer is null]\n", job.literal_offset, job.n_literals);
        return;
    }

    // Calculate the actual pointer
    Lit* job_literals_ptr = literal_buffer_base_ptr + job.literal_offset;

#ifdef __CUDA_ARCH__
    // Device-side printing
    printf("Job (offset %zu): ", job.literal_offset);
    for (size_t i = 0; i < job.n_literals; i++) {
        print_lit(job_literals_ptr[i]); // Use calculated pointer
        printf(" ");
    }
    printf("\n");
#else
    // Host-side printing
    Lit* literals_host = new Lit[job.n_literals];

    // Copy from device using the calculated pointer
    check(cudaMemcpy(literals_host, job_literals_ptr,
              sizeof(Lit) * job.n_literals, cudaMemcpyDeviceToHost),
        "Copying job literals to host for printing");

    printf("Job (offset %zu): ", job.literal_offset);
    for (size_t i = 0; i < job.n_literals; i++) {
        print_lit(literals_host[i]);
        printf(" ");
    }
    printf("\n");

    delete[] literals_host;
#endif
}

// Updated cleanup - explicitly call destroy() on GPUVec members
__host__ void JobsQueue::free_job_literals()
{
    printf("Cleaning up JobsQueue...\n");
    // Explicitly destroy GPUVec members as they don't have automatic destructors
    all_literals_buffer.destroy();
    jobs.destroy();
    printf("JobsQueue cleanup complete.\n");
}
