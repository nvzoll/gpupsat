#ifndef SAT_SOLVER_CUH_
#define SAT_SOLVER_CUH_

#include "Configs.cuh"

#include "Utils/CUDAClauseVec.cuh"
#include "Utils/GPUVec.cuh"
#include "SolverTypes.cuh"
#include "ConflictAnalysis/ConflictAnalyzer.cuh"
#include "ConflictAnalysis/ConflictAnalyzerWithWatchedLits.cuh"
#include "ConflictAnalysis/ConflictAnalyzerFullSearchSpace.cuh"
#include "Utils/NodesRepository.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "VariablesStateHandler.cuh"
#include "DecisionMaker.cuh"
#include "Statistics/RuntimeStatistics.cuh"
#include "Restarts/GeometricRestartsManager.cuh"

class SATSolver
{

private:
    const CUDAClauseVec *formula;

    DecisionMaker decision_maker;
    /**
     * The three vectors below store literals (or decisions that hold literals), which
     * indicate the values for the vars of the formula in the current stage of the
     * solutions problem, since a var can only be assign one value, these
     * three vectors MUST ALWAYS (once the constructor is run):
     *         * Be disjoint
     *         * Together contain literals for every var that is not in
     *            free_vars (any non-free var)
     *
     * Some methods in this class assume those requirements and may not work
     * properly if they are not met.
     * The three vectors are:
     *       * decisions: The decisions made so far, only one decision is made at each
     *          decision level.
     *       * implications: The implications of each decision, as determined by BCP
     *          (Boolean Constraint Propagation). They are intrinsically connected with
     *          the decision that generated them and MUST be undone once the decision is
     *          undone.
     *       * assumptions: Assumptions made before starting the solving process. These
     *          assumptions work as decisions, but they are never undone. If no solution
     *          is found with the assumptions, even if the formula itself is sat_status::SAT, the
     *          procedure returns sat_status::UNSAT.
     */
    VariablesStateHandler vars_handler;

    ////////

    int n_vars;
    sat_status current_status;

#ifdef USE_RESTART
    GeometricRestartsManager geometric_restart;
#endif

#if CONFLICT_ANALYSIS_STRATEGY == TWO_WATCHED_LITERALS
    ConflictAnalyzerWithWatchedLits conflictAnalyzer;
#endif

#if CONFLICT_ANALYSIS_STRATEGY == FULL_SPACE_SEARCH
    ConflictAnalyzerFullSearchSpace conflictAnalyzer;
#endif

#if CONFLICT_ANALYSIS_STRATEGY == BASIC_SEARCH
    ConflictAnalyzer conflictAnalyzer;
#endif

    Decision next_decision;

    RuntimeStatistics *stats;

    /**
     * return: sat_status::SAT if the formula has been satisfied, unsat if there was a conflict
     * undef if it is still not solved.
     */
    __device__ sat_status propagate(Decision d);
    //__device__ void backtrack(int decision_level);
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ sat_status preprocess(GPUVec<Lit> *assumptions);
#else
    __device__ sat_status preprocess(GPUStaticVec<Lit> *assumptions);
#endif

    __device__ void restart();


public:
    __device__ SATSolver(const CUDAClauseVec *formula,
                         int n_vars,
                         int max_implication_per_var,
                         /**
                          * Variables to be ignored (specially if they are already solved).
                          * Should not be contained on the assumptions.
                          */
                         const GPUVec<Var> *dead_vars,
                         RuntimeStatistics *statistics,
                         watched_clause_node_t *node_repository
                         //,GPUVec<WatchedClause> & watched_clauses
                        );

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ sat_status solve(GPUVec<Lit> *assumptions);
#else
    __device__ sat_status solve(GPUStaticVec<Lit> *assumptions);
#endif
    __device__ sat_status solve();
    /**
     * Stores the results on results(if sat_status::SAT).
     */
    __device__ void get_results(Lit *results);
    __device__ size_t get_results_size();
    __device__ sat_status clause_status(Clause *c);
    __device__ void reset();

    __device__ void print_structures();
};

#endif /* SAT_SOLVER_CUH_ */
