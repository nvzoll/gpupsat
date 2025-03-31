// mkJob_dev.cu

#include "mkJob_dev.cuh"
#include "./JobsQueue.cuh"
#include "ErrorHandler/CudaMemoryErrorHandler.cuh"
#include <vector>

__host__ __device__ Job *mkJob_dev(JobsQueue *queue, const std::vector<Lit> &lits)
{
    Job job;
    job.size = lits.size();

    // Check if we can use the pre-allocated buffer
    if (queue->get_literal_buffer_ptr() != nullptr)
    {
        Lit *literals_mem = queue->reserve_literal_space(lits.size());
        if (literals_mem != nullptr)
        {
            // Use the pre-allocated space
            for (size_t i = 0; i < lits.size(); i++)
            {
                literals_mem[i] = lits[i];
            }
            job.literals = literals_mem;
        }
        else
        {
            // Fall back to individual allocation if buffer is full
            Lit *literals_mem = nullptr;
#ifdef __CUDA_ARCH__
            // Device code path
            literals_mem = (Lit *)malloc(lits.size() * sizeof(Lit));
#else
            // Host code path
            check(cudaMalloc(&literals_mem, lits.size() * sizeof(Lit)), "Allocating memory for job literals");
#endif
            for (size_t i = 0; i < lits.size(); i++)
            {
                literals_mem[i] = lits[i];
            }
            job.literals = literals_mem;
        }
    }
    else
    {
        // No pre-allocated buffer, use existing allocation
        Lit *literals_mem = nullptr;
#ifdef __CUDA_ARCH__
        // Device code path
        literals_mem = (Lit *)malloc(lits.size() * sizeof(Lit));
#else
        // Host code path
        check(cudaMalloc(&literals_mem, lits.size() * sizeof(Lit)), "Allocating memory for job literals");
#endif
        for (size_t i = 0; i < lits.size(); i++)
        {
            literals_mem[i] = lits[i];
        }
        job.literals = literals_mem;
    }

    queue->add_job(job);
    return queue->get_last_job_ptr();
}