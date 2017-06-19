#ifndef __UIPFINDER_CUH__
#define __UIPFINDER_CUH__

#include "CUDAListGraph.cuh"
#include "Utils/GPUStaticVec.cuh"

class UIPFinder
{
private:
    CUDAListGraph *impl_graph;
    int n_vars;
    GPUStaticVec<Var, 200> var_in_paths;

    __device__ void paths_to_decision(Decision target_decision, Var current_var,
                                      GPUStaticVec<Var, 200> *current_path, bool *first_path);

public:
    __device__ UIPFinder(CUDAListGraph *impl_graph, int n_vars);
    __device__ Var first_uip(Decision target_decision);
};

#endif /* __UIPFINDER_CUH__ */
