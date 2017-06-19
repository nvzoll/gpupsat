#ifndef __BACKTRACKER_CUH__
#define __BACKTRACKER_CUH__

#include "VariablesStateHandler.cuh"
#include "BCPStrategy/WatchedClausesList.cuh"
#include "ConflictAnalysis/CUDAListGraph.cuh"
#include "Statistics/RuntimeStatistics.cuh"

class Backtracker
{
private:
    VariablesStateHandler *vars_handler;
    WatchedClausesList *watched_clauses;
    CUDAListGraph *graph;
    RuntimeStatistics *stats;

    __device__ void backtrack_to(int decision_level);

public:
    __device__ Backtracker(VariablesStateHandler *handler,
                           WatchedClausesList *watched_clauses,
                           CUDAListGraph *graph,
                           RuntimeStatistics *stats);

    __device__ sat_status handle_backtrack(int decision_level, bool switch_literal);
    __device__ sat_status handle_backtrack(int decision_level);
    __device__ void set_watched_clauses(WatchedClausesList *watched_clause);

    __device__ sat_status handle_chronological_backtrack();
};

#endif /* __BACKTRACKER_CUH__ */
