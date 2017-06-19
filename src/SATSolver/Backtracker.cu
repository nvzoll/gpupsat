#include "Backtracker.cuh"

__device__ Backtracker::Backtracker(
    VariablesStateHandler *handler,
    WatchedClausesList *watched_clauses,
    CUDAListGraph *graph,
    RuntimeStatistics *stats)

    : vars_handler { handler }
    , watched_clauses { watched_clauses }
    , graph { graph }
    , stats { stats }
{
#ifdef USE_ASSERTIONS
    assert(handler);
#endif
}

__device__ void Backtracker::backtrack_to(int decision_level)
{
    stats->signal_backtrack_start(vars_handler->n_decisions());

    vars_handler->backtrack_to(decision_level);

    if (graph != nullptr) {
        graph->backtrack_to(decision_level);
    }

    if (watched_clauses != nullptr) {
        watched_clauses->handle_backtrack();
    }

    stats->signal_backtrack_stop(vars_handler->n_decisions());
}

__device__ sat_status Backtracker::handle_backtrack(int decision_level)
{
    return handle_backtrack(decision_level, true);
}

__device__ sat_status Backtracker::handle_backtrack(int decision_level, bool switch_literal)
{

    Decision to_backtrack =
        vars_handler->last_non_branched_decision(decision_level);

    to_backtrack.branched = true;
    to_backtrack.literal = ~to_backtrack.literal;


    //if (decision_level < 0 || vars_handler->get_decision_level() == -1)
    if (to_backtrack.decision_level == NULL_DECISION_LEVEL) {
        backtrack_to(0);
        return sat_status::UNSAT;
    }

    if (switch_literal) {
        backtrack_to(to_backtrack.decision_level - 1);
        vars_handler->new_decision(to_backtrack);
        vars_handler->set_decision_level(vars_handler->get_decision_level() + 1);
        if (graph != nullptr) {
            Decision implication = to_backtrack;
            implication.implicated_from_formula = false;
            graph->set(implication);
        }
    }
    else {
        if (to_backtrack.decision_level == decision_level) {
            backtrack_to(to_backtrack.decision_level - 1);
        }
        else {
            backtrack_to(to_backtrack.decision_level);
        }
    }

    return sat_status::UNDEF;

}

__device__ sat_status Backtracker::handle_chronological_backtrack()
{
    return handle_backtrack(vars_handler->get_decision_level());
}

__device__ void Backtracker::set_watched_clauses(WatchedClausesList *watched_clause)
{
    this->watched_clauses = watched_clause;
}
