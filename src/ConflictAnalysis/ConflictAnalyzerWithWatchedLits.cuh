#ifndef __CONFLICTANALYZERWITHWATCHEDLITS_CUH__
#define __CONFLICTANALYZERWITHWATCHEDLITS_CUH__

#include "Utils/CUDAClauseVec.cuh"
#include "Utils/GPUVec.cuh"
#include "ConflictAnalyzer.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "BCPStrategy/WatchedClausesList.cuh"
#include "Utils/GPUStaticVec.cuh"

class ConflictAnalyzerWithWatchedLits : public ConflictAnalyzer
{
public:
    __device__ ConflictAnalyzerWithWatchedLits(
            int n_var,
            const CUDAClauseVec *formula,
            VariablesStateHandler *handler,
            bool use_implication_graph,
            int max_implication_per_var,
            DecisionMaker *dec_maker,
            RuntimeStatistics *stats,
            watched_clause_node_t *node_repository
            /*,GPUVec <WatchedClause> & watched_clauses_repository*/);

    __device__ void reset();
    __device__ void restart();
    __device__ int get_n_last_conflicts();
    __device__ sat_status propagate(Decision d);
    __device__ sat_status handle_conflict_with_clause_learning(Clause *c);
    __device__ sat_status handle_learnt_clause(Clause *learnt,
            Decision highest, Decision second_highest);

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

    // Test methods
    __device__ void print_two_watched_literals_structure();
    __device__ bool check_two_watched_literals_consistency();
    __device__ void print_graph();

private:
    WatchedClausesList watched_clauses;

    /**
     *
     */
    __device__ sat_status propagate_all_clauses(Decision d, const Clause **conflicting_clause);

};

#endif /* __CONFLICTANALYZERWITHWATCHEDLITS_CUH__ */

