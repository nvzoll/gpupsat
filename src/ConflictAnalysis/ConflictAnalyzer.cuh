#ifndef __CONFLICTANALYZER_CUH__
#define __CONFLICTANALYZER_CUH__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "CUDAListGraph.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "Utils/CUDAClauseVec.cuh"
#include "GraphAnalyzer.cuh"
#include "SATSolver/Configs.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "SATSolver/Configs.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "BCPStrategy/WatchedClausesList.cuh"
#include "SATSolver/Backtracker.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "ClauseLearning/LearntClausesManager.cuh"
#include "SATSolver/DecisionMaker.cuh"

class ConflictAnalyzer
{
public:
    __device__ ConflictAnalyzer(int n_var,
                                const CUDAClauseVec *formula,
                                VariablesStateHandler *,
                                bool use_implication_graph,
                                int max_implication_per_var,
                                DecisionMaker *,
                                RuntimeStatistics *);

    __device__ void reset();
    __device__ sat_status propagate(Decision d);
    /**
     * Backtracks the current implication graph to an specific decision level, removing
     * vertices and edges that are smaller than the specified decision level.
     * * Does NOT remove implications or decisions!
     */
    //__device__ void backtrack_to(int decision_level);
    /**
     * Returns the status after the assumptions, if they are processed.
     */
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ status set_assumptions(GPUVec<Lit> *assumptions);
#else
    __device__ sat_status set_assumptions(GPUStaticVec<Lit> *assumptions);
#endif
    __device__ void make_decision(Decision decision);
    __device__ void unmake_implication(Decision implication);
    __device__ Clause learn_clause(Decision& next_dec_level,
                                   Decision& highest);
    __device__ void add_decision_to_graph(Decision decision);

    __device__ bool is_set_in_graph(Var var);

    __device__ bool get_use_implication_graph();
    __device__ void restart();
    /**
     * This returns a number that indicates how many conflicts happened
     * when the last call of "propagate" was made.
     */
    __device__ int get_n_last_conflicts();
    // Test methods:
    __device__ bool check_consistency();
    __device__ bool check_solver_consistency();
    __device__ void print_graph();


protected:
    CUDAListGraph graph;
    Backtracker backtracker;

    int n_var;
    int max_impl_per_var;
    const CUDAClauseVec *formula;

    VariablesStateHandler *vars_handler;
    DecisionMaker *decision_maker;
    int n_last_conflicts;

    RuntimeStatistics *stats;

    bool use_implication_graph;

#ifdef USE_CLAUSE_LEARNING
    LearntClausesManager learnt_clauses_manager;
#endif

    __device__ void add_implication(Decision implication, const Clause *clause);
    __device__ inline sat_status clause_status(const Clause *c, bool implicated_from_formula);
    __device__ void add_conflict_to_graph(const Clause *conflicting_clause);
    __device__ sat_status handle_conflict(const Clause *c, Decision& implicated);
    __device__ sat_status handle_conflict_without_clause_learning();
    __device__ sat_status handle_conflict_with_clause_learning(const Clause *c, Decision& implicated);
    __device__ void handle_learnt_clause(const Clause *learnt, Decision highest, Decision second_highest);
    __device__ sat_status implicate_and_backtrack_from_learnt_clause(
        Clause const& learnt_clause, Decision highest, Decision second_highest, Decision& implicated);

    __device__ virtual sat_status propagate_all_clauses(Decision d, const Clause **conflicting_clause);
    __device__ sat_status process_unit_clauses();

};

#endif /* __CONFLICTANALYZER_CUH__ */
