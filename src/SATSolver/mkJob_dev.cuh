#ifndef MK_JOB_DEV_CUH
#define MK_JOB_DEV_CUH

#include <vector>
#include "SATSolver/SolverTypes.cuh"

// Forward declaration of JobsQueue class
class JobsQueue;

/**
 * Creates a job with the given literals and adds it to the queue
 *
 * @param queue The job queue where the job will be added
 * @param lits The literals to be included in the job
 * @return Pointer to the created job in the queue
 */
__host__ __device__ Job *mkJob_dev(JobsQueue *queue, const std::vector<Lit> &lits);

#endif // MK_JOB_DEV_CUH
