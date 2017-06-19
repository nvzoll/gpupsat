#include "CUDAListGraph.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "SATSolver/Configs.cuh"

__device__ Clause analyze_graph(CUDAListGraph& graph, Decision& second_highest,
                                Decision& highest);

