#include "ConflictAnalyzer.cuh"
#include "ConflictAnalyzerFullSearchSpace.cuh"

__device__ ConflictAnalyzerFullSearchSpace::ConflictAnalyzerFullSearchSpace(
    int n_var,
    const CUDAClauseVec *formula,
    VariablesStateHandler *handler,
    int max_implication_per_var,
    DecisionMaker *dec_maker,
    RuntimeStatistics *stats)
    : ConflictAnalyzer(n_var,
                       formula, handler,
                       false,
                       max_implication_per_var,
                       dec_maker,
                       stats)
{
#ifdef USE_CLAUSE_LEARNING
    learnt_clauses_manager.set_watched_clauses(nullptr);
#endif
}

__device__ void ConflictAnalyzerFullSearchSpace::reset()
{
    ConflictAnalyzer::reset();
}
__device__ sat_status ConflictAnalyzerFullSearchSpace::propagate(Decision d)
{
    if (!(vars_handler->no_free_vars())) {
        return sat_status::UNDEF;
    }

    sat_status status =  ConflictAnalyzer::propagate(d);

    return status;
}

__device__ sat_status ConflictAnalyzerFullSearchSpace::propagate_all_clauses(Decision d, const Clause **conflicting)
{
    return ConflictAnalyzer::propagate_all_clauses(d, conflicting);
}
/**
 * Backtracks the current implication graph to an specific decision level, removing
 * vertices and edges that are smaller than the specified decision level.
 * * Does NOT remove implications or decisions!
 */
/*
__device__ void ConflictAnalyzerFullSearchSpace::backtrack_to(int decision_level)
{
    ConflictAnalyzer::backtrack_to(decision_level);
}
*/
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ sat_status ConflictAnalyzerFullSearchSpace::set_assumptions(GPUVec<Lit> *assumptions)
#else
    __device__ sat_status ConflictAnalyzerFullSearchSpace::set_assumptions(GPUStaticVec<Lit> *assumptions)
#endif
{
    return ConflictAnalyzer::set_assumptions(assumptions);
}
__device__ void ConflictAnalyzerFullSearchSpace::restart()
{
    ConflictAnalyzer::restart();
}
__device__ int ConflictAnalyzerFullSearchSpace::get_n_last_conflicts()
{
    return ConflictAnalyzer::get_n_last_conflicts();
}
