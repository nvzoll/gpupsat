#ifndef __PARALLELIZER_CUH__
#define __PARALLELIZER_CUH__

#include "Configs.cuh"
#include "DataToDevice.cuh"
#include "JobsQueue.cuh"
#include "Results.cuh"
#include "SATSolver.cuh"
#include "SolverTypes.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "Utils/GPUStaticVec.cuh"

struct KernelContext;

// Flat array of inline-stored KernelContexts. Allocated by allocate_kernel_contexts; constructed
// in-place by parallel_kernel_init.
struct KernelContextStorage {
    KernelContext *data;
};

__host__ void allocate_kernel_contexts(KernelContextStorage *storage, int n_blocks, int n_threads);
__host__ void free_kernel_contexts(KernelContextStorage *storage);

__global__ void parallel_kernel_init(DataToDevice *data, KernelContextStorage);
__global__ void parallel_kernel(KernelContextStorage, int *state);
__global__ void parallel_kernel_retrieve_results(DataToDevice data, KernelContextStorage, Lit *res_buf);

__global__ void run_sequential(DataToDevice data, int *state, Lit *res_buf);

#endif /* __PARALLELIZER_CUH__ */
