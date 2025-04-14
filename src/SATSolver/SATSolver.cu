#include "SATSolver.cuh"

__device__ SATSolver::SATSolver(
    const CUDAClauseVec *formula,
    int n_vars,
    int max_implication_per_var,
    const GPUVec<Var> *dead_vars,
    RuntimeStatistics *statistics,
    watched_clause_node_t *node_repository
    /*, GPUVec<WatchedClause> & watched_clauses*/)

    : decision_maker(&vars_handler, formula, n_vars)
    , vars_handler(n_vars, dead_vars, &decision_maker)

    , conflictAnalyzer(n_vars, formula, &vars_handler,
#if CONFLICT_ANALYSIS_STRATEGY != FULL_SPACE_SEARCH
#ifdef USE_CONFLICT_ANALYSIS
                       true,
#else
                       false,
#endif
#endif
                       max_implication_per_var, &decision_maker
                       , statistics
#if CONFLICT_ANALYSIS_STRATEGY == TWO_WATCHED_LITERALS
                       , node_repository
                       //, watched_clauses
#endif
                      )

#ifdef  USE_RESTART
    , geometric_restart(GEOMETRIC_CONFLICTS_BEFORE_RESTART, GEOMETRIC_RESTART_INCREASE_FACTOR)
#endif
#ifdef USE_RESTART
#endif

    , formula { formula }
    //, decision_level { 0 }
    , n_vars { n_vars }
    , current_status { sat_status::UNDEF }
    , stats { statistics }
{
    next_decision.decision_level = NULL_DECISION_LEVEL;
}

__device__ void SATSolver::restart()
{
#ifdef USE_RESTART
    conflictAnalyzer.restart();
#endif
}

/**
 * Solve with no assumptions.
 */
__device__ sat_status SATSolver::solve()
{
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> no_assumptions(0);
#else
    GPUStaticVec<Lit> no_assumptions;
#endif
    sat_status status = solve(&no_assumptions);
    vars_handler.set_assumptions(nullptr);

    return status;
}

/**
 * Solve using assumptions.
 */
__device__ sat_status SATSolver::solve(
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> *assumptions
#else
    GPUStaticVec<Lit> *assumptions
#endif
)
{
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Block %d, thread %d is starting to solve...\n", blockIdx.x, threadIdx.x);
    }

    stats->signal_preprocess_start();

    sat_status status_post_assumptions = preprocess(assumptions);

    stats->signal_preprocess_stop();

    if (status_post_assumptions == sat_status::SAT) {
        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("sat_status::SAT on pre-processing!\n");
            vars_handler.print_assumptions();
            vars_handler.print_decisions();
            vars_handler.print_implications();
        }

        current_status = sat_status::SAT;
        return sat_status::SAT;
    }

    if (status_post_assumptions == sat_status::UNSAT) {
        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            vars_handler.print_assumptions();
            vars_handler.print_decisions();
            vars_handler.print_implications();
            printf("sat_status::UNSAT on pre-processing!\n");
        }

        current_status = sat_status::UNSAT;
        return sat_status::UNSAT;
    }

    sat_status status;

    int n_iterations = 0;

#ifdef USE_ASSERTIONS
    assert(!vars_handler.no_free_vars());
#endif

#ifdef IMPLICATION_GRAPH_DEBUG
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("DEBUG (%d, %d): starting loop.\n", blockIdx.x, threadIdx.x);
    }
#endif

    while (true) {
        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("DEBUG (%d, %d): implication graph on iteration %d is:\n", blockIdx.x, threadIdx.x, n_iterations);
            //conflictAnalyzer.print_graph();
            vars_handler.print_decisions();
            vars_handler.print_implications();
            vars_handler.print_assumptions();
        }

        stats->signal_decision_start();
        bool decision_status =
            decision_maker.decide().decision_level != NULL_DECISION_LEVEL;//decide();
        stats->signal_decision_stop();

        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("(after decision) ");
            vars_handler.print_decisions();
            vars_handler.print_assumptions();
        }

        if (!decision_status) {
            status = sat_status::UNSAT;

            if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
                printf("Decision maker returned false!\n");
            }

            break;
        }

        stats->signal_conflict_analysis_start(vars_handler.get_decision_level());

        status = propagate(vars_handler.get_last_decision());

        stats->signal_conflict_analysis_stop(vars_handler.get_decision_level());

        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("Propagate returned status:");
            print_status(status);
            printf("\n");
        }

        if (status != sat_status::UNDEF) {
            break;
        }

        if (vars_handler.no_free_vars()) {
            status = sat_status::SAT;
            break;
        }

#ifdef USE_RESTART
        for (int i = 0; i < conflictAnalyzer.get_n_last_conflicts(); i++) {
            geometric_restart.signal_conflict();
        }

        if (geometric_restart.should_restart()) {
            restart();
        }
#endif

        n_iterations++;

#ifdef MAX_ITERATIONS
        if (n_iterations > MAX_ITERATIONS) {
            printf("Limit of iterations has been reached. No answer was found.\n");
            current_status = sat_status::UNDEF;
            return sat_status::UNDEF;
        }
#endif

    }

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Solved: ");
        print_status(status);
        printf("\n");
        vars_handler.print_all();
    }

    if (status == sat_status::SAT) {
        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("It seems that the formula is sat_status::SAT!\n");
        }

        current_status = sat_status::SAT;
    }

    if (status == sat_status::UNSAT || status == sat_status::UNDEF) {
        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("The formula seems to be sat_status::UNSAT!\n");
        }
        current_status = sat_status::UNSAT;
    }

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Solving is done.\n");
    }

    return current_status;
}

/**
 * Propagates the currents decisions and implications, possibly generating more
 * implications (BCP through UP). Returns if the decisions have satisfied, unsatisfied or not
 * changed the current status of the formula.
 */
__device__ sat_status SATSolver::propagate(Decision d)
{
    return conflictAnalyzer.propagate(d);
}

/**
 * This method sets the assumptions, remove the assumptions from the free vars and
 * propagate the assumptions.
 */
__device__ sat_status SATSolver::preprocess(
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> *assumptions
#else
    GPUStaticVec<Lit> *assumptions
#endif
)
{
    stats->signal_pre_proc_handling_vars_start();

    vars_handler.set_assumptions(assumptions);

    stats->signal_pre_proc_handling_vars_stop();

    sat_status status_post_preprocessing =
        conflictAnalyzer.set_assumptions(assumptions);

    //vars_handler.print_assumptions();

    if (status_post_preprocessing != sat_status::UNDEF) {
        return status_post_preprocessing;
    }

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Starting to pre-process. Assumptions:\n");
        vars_handler.print_assumptions();
    }

    if (vars_handler.no_assumptions()
        && vars_handler.no_implications()
        && vars_handler.no_decisions()) {
        return sat_status::UNDEF;
    }

    if (status_post_preprocessing == sat_status::UNDEF &&
        vars_handler.no_free_vars()) {
        status_post_preprocessing = sat_status::SAT;
    }

    // if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        //printf("Preprocessing has generated this implication graph:\n");
        //conflictAnalyzer.print_graph();
    // }

    return status_post_preprocessing;
}

__device__ void SATSolver::get_results(Lit *results)
{
    if (current_status != sat_status::SAT) {
        return;
    }
    else {
        int next_pos = 0;

        for (int i = 0; i < vars_handler.n_decisions(); i++) {
            results[next_pos] = vars_handler.get_decision(i).literal;
            next_pos++;
        }

        if (vars_handler.assumptions_set()) {
            for (int i = 0; i < vars_handler.n_assumptions(); i++) {
                results[next_pos] = vars_handler.get_assumption(i);
                next_pos++;
            }
        }
        for (int i = 0; i < vars_handler.n_implications(); i++) {
            results[next_pos] = vars_handler.get_implication(i)->literal;
            next_pos++;
        }
    }
}

__device__ size_t SATSolver::get_results_size()
{
    size_t size = vars_handler.n_decisions()
                + vars_handler.n_implications();

    size += vars_handler.assumptions_set() ? vars_handler.n_assumptions() : 0;

    return size;
}

/**
 * Resets the sat_status::SAT Solver, allowing to used it again.
 * Does not delete the learnt clauses (when implemented).
 */
__device__ void SATSolver::reset()
{
    vars_handler.reset();
    current_status = sat_status::UNDEF;
    conflictAnalyzer.reset();
}

__device__ void SATSolver::print_structures()
{
    conflictAnalyzer.print_graph();
    vars_handler.print_assumptions();
    vars_handler.print_decisions();
    vars_handler.print_implications();
}
