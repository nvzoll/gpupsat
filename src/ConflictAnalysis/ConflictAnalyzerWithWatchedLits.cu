#include "ConflictAnalyzerWithWatchedLits.cuh"

__device__ ConflictAnalyzerWithWatchedLits::ConflictAnalyzerWithWatchedLits(
    int n_var,
    const CUDAClauseVec *formula,
    VariablesStateHandler *handler,
    bool use_implication_graph,
    int max_implication_per_var,
    DecisionMaker *dec_maker,
    RuntimeStatistics *stats,
    watched_clause_node_t *node_repository
    /*,GPUVec <WatchedClause> & watched_clauses_repository*/)
    : ConflictAnalyzer(n_var, formula, handler, use_implication_graph,
                       max_implication_per_var,
                       dec_maker,
                       stats)

    , watched_clauses(n_var, handler,
                      node_repository,
                      // &watched_clauses_repository,
                      formula->size_of())
{
    backtracker.set_watched_clauses(&watched_clauses);
    watched_clauses.add_all_clauses(*formula);

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Two-watched literals before starting:\n");
        watched_clauses.print_structure();
    }

#ifdef USE_CLAUSE_LEARNING
    learnt_clauses_manager.set_watched_clauses(&watched_clauses);
#endif
}

__device__ void ConflictAnalyzerWithWatchedLits::reset()
{
    ConflictAnalyzer::reset();
    watched_clauses.reset();
}
__device__ sat_status ConflictAnalyzerWithWatchedLits::propagate(Decision d)
{
    return ConflictAnalyzer::propagate(d);
}

#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    __device__ sat_status ConflictAnalyzerWithWatchedLits::set_assumptions(GPUVec<Lit> *assumptions)
#else
    __device__ sat_status ConflictAnalyzerWithWatchedLits::set_assumptions(GPUStaticVec<Lit> *assumptions)
#endif
{
    stats->signal_pre_proc_handle_assumptions_start();

    GPULinkedList<found_implication> new_implications;

    for (int i = 0; i < assumptions->size_of(); i++) {
        Decision d;
        d.decision_level = 0;
        d.literal = assumptions->get(i);
        d.implicated_from_formula = false;

        sat_status current_status = vars_handler->literal_status(d.literal);

        const Clause *conflicting;
        sat_status status = watched_clauses.new_decision(d, &conflicting,
                            new_implications);

        if (status != sat_status::UNDEF) {
            stats->signal_pre_proc_handle_assumptions_stop();
            return status;
        }

    }

    stats->signal_pre_proc_handle_assumptions_stop();
    stats->signal_pre_proc_add_to_graph_start();

    if (use_implication_graph) {
        for (int i = 0; i < vars_handler->n_assumptions(); i++) {
            Decision d;
            d.literal = vars_handler->get_assumption(i);
            d.decision_level = 0;
            d.implicated_from_formula = false;
            graph.set(d);
        }
        GPULinkedList<found_implication>::LinkedListIterator iter =
            new_implications.get_iterator();

        while (iter.has_next()) {
            found_implication impl = iter.get_next();
            ConflictAnalyzer::add_implication(impl.implication,
                                              impl.implicating_clause);
        }

    }
    else {
        vars_handler->add_many_implications(new_implications);
    }

    new_implications.unalloc();

    stats->signal_pre_proc_add_to_graph_stop();

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("After adding assumptions:\n");
        watched_clauses.print_structure();
        vars_handler->print_implications();
    }

    return sat_status::UNDEF;
}

__device__ sat_status ConflictAnalyzerWithWatchedLits::propagate_all_clauses(Decision d,
        const Clause **conflicting_clause)
{
    GPULinkedList<found_implication> new_implications;
    sat_status stat = watched_clauses.new_decision(d, conflicting_clause,
                      new_implications);

    GPULinkedList<found_implication>::LinkedListIterator iter =
        new_implications.get_iterator();


    if (use_implication_graph) {
        while (iter.has_next()) {
            found_implication impl = iter.get_next();
            ConflictAnalyzer::add_implication(impl.implication,
                                              impl.implicating_clause);
        }
    }
    else {
        vars_handler->add_many_implications(new_implications);
    }

    new_implications.unalloc();

    //watched_clauses.print_structure();
    return stat;
}

__device__ void ConflictAnalyzerWithWatchedLits::restart()
{
    ConflictAnalyzer::restart();
}

__device__ int ConflictAnalyzerWithWatchedLits::get_n_last_conflicts()
{
    return ConflictAnalyzer::get_n_last_conflicts();
}

__device__ void ConflictAnalyzerWithWatchedLits::print_two_watched_literals_structure()
{
    watched_clauses.print_structure();
}

__device__ bool ConflictAnalyzerWithWatchedLits::check_two_watched_literals_consistency()
{
    return watched_clauses.check_consistency();
}
__device__ void ConflictAnalyzerWithWatchedLits::print_graph()
{
    ConflictAnalyzer::print_graph();
}
