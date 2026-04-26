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

struct KernelContextStorage {
    void **data;
    size_t pitch;
};

__global__ void parallel_kernel_init(DataToDevice *data, KernelContextStorage);
__global__ void parallel_kernel(KernelContextStorage, int *state);
__global__ void parallel_kernel_retrieve_results(DataToDevice data, KernelContextStorage);

__global__ void run_sequential(DataToDevice data, int *state);

#endif /* __PARALLELIZER_CUH__ */
