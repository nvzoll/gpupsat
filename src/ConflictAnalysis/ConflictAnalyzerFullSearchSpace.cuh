#ifndef __CONFLICTANALYZERFULLSEARCHSPACE_CUH__
#define __CONFLICTANALYZERFULLSEARCHSPACE_CUH__

class ConflictAnalyzerFullSearchSpace : public ConflictAnalyzer
{
public:
    __device__ ConflictAnalyzerFullSearchSpace(
        int n_var,
        const CUDAClauseVec *formula,
        VariablesStateHandler *handler,
        int max_implication_per_var,
        DecisionMaker *dec_maker,
        RuntimeStatistics *stats);

    __device__ void reset();
    __device__ int get_n_last_conflicts();
    __device__ sat_status propagate(Decision d);
    /**
     * Backtracks the current implication graph to an specific decision level, removing
     * vertices and edges that are smaller than the specified decision level.
     * * Does NOT remove implications or decisions!
     */
    //__device__ void backtrack_to(int decision_level);
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ sat_status set_assumptions(GPUVec<Lit> *assumptions);
#else
    __device__ sat_status set_assumptions(GPUStaticVec<Lit> *assumptions);
#endif
    __device__ void restart();
private:
    __device__ sat_status propagate_all_clauses(Decision d, const Clause **conflicting);
};

#endif /* __CONFLICTANALYZERFULLSEARCHSPACE_CUH__ */
