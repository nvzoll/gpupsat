#ifndef __PARALLELIZER_CUH__
#define __PARALLELIZER_CUH__

#include "Utils/CUDAClauseVec.cuh"
#include "SATSolver.cuh"
#include "SolverTypes.cuh"
#include "JobsQueue.cuh"
#include "Configs.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "Results.cuh"
#include "DataToDevice.cuh"

struct KernelContextStorage
{
    void **data;
    size_t pitch;
};

// POD struct to safely pass data pointers/values to kernels
struct KernelDataPod
{
    RuntimeStatistics *statistics_ptr;
    JobsQueue *queue_ptr;
    const CUDAClauseVec *clauses_db_ptr;
    // Replace pointer to GPUVec object with raw pointer and size
    const Var *dead_vars_elements_ptr;
    size_t dead_vars_size;
    NodesRepository<GPULinkedList<WatchedClause *>::Node> *nodes_repository_ptr;
    unsigned int *found_answer_ptr;
    Results *results_ptr;

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<GPUVec<Lit>> *all_assumptions_parallel_ptr;
#endif

    int n_vars;
    int max_implication_per_var;

    // Added pointer to the base of the shared literal buffer
    Lit *literal_buffer_base_ptr;
};

__global__ void parallel_kernel_init(KernelDataPod *pod_data, KernelContextStorage); // Signature changed
__global__ void parallel_kernel(KernelContextStorage, int *state);                   // Might need pod_data too later
// __global__ void parallel_kernel_retrieve_results(KernelDataPod pod, KernelContextStorage); // Removed this kernel

__global__ void run_sequential(DataToDevice data, int *state); // Keep as is for now, assuming sequential path is separate

#endif /* __PARALLELIZER_CUH__ */
