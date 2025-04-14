#include "WatchedClausesList.cuh"

__device__ WatchedClausesList::WatchedClausesList(int n_vars,
        VariablesStateHandler *handler,
        watched_clause_node_t *node_repository,
        //GPUVec< WatchedClause > * repository,
        int n_clauses
                                                 ) :
    structure(n_vars, handler, node_repository)
{
    this->vars_handler = handler;
    this->next_watched_clause_index = 0;
    this->repository = new GPUVec<WatchedClause>(n_clauses + 1
#ifdef USE_CLAUSE_LEARNING
            + MAX_LEARNT_CLAUSES_PER_THREAD
#endif
                                                ); //repository;
    for (int i = 0; i < n_clauses + 1
#ifdef USE_CLAUSE_LEARNING
         + MAX_LEARNT_CLAUSES_PER_THREAD
#endif
         ; i++) {
        WatchedClause wc;
        //wc.clause = nullptr;
        wc.watched_lit_index_1 = -1;
        wc.watched_lit_index_2 = -1;
        this->repository->add(wc);
    }
}

__device__ void WatchedClausesList::replace_clause(Clause old, Clause clause_to_add)
{

    structure.purge_clause(old);

    for (int i = repository->size_of() - 1; i >= 0; i--) {
        WatchedClause *wc = repository->get_ptr(i);
        if (wc->clause == old) {
            Lit dummy;
            new_clause(clause_to_add, wc, dummy, true);
            return;
        }
    }
}

__device__ sat_status WatchedClausesList::new_decision(Decision decision,
        const Clause **conflicting_clause,
        GPULinkedList<found_implication>& implication_list)
{

    int last_implications_size = implication_list.size();

    GPULinkedList<WatchedClause *>::LinkedListIterator iter =
        structure.get_iterator(var(decision.literal));

    GPULinkedList<WatchedClause *> temporary_clauses;

    while (iter.has_next()) {
        WatchedClause *wc = iter.get_next();
        temporary_clauses.push_back(wc);
    }

    GPULinkedList<WatchedClause *>::LinkedListIterator temp_iter =
        temporary_clauses.get_iterator();

    while (temp_iter.has_next()) {
        WatchedClause *wc = temp_iter.get_next();

        bool conflict = process_clause(wc, decision, implication_list);

        if (conflict) {
            *conflicting_clause = &(wc->clause);
            return sat_status::UNSAT;
        }

    }
    temporary_clauses.unalloc();

    int implications_count = implication_list.size() - last_implications_size;

    GPULinkedList<found_implication>::LinkedListIterator iter2 =
        implication_list.get_iterator(last_implications_size);

    for (int i = implications_count; i > 0; i--) {
#ifdef USE_ASSERTIONS

        assert(iter2.has_next());
#endif


        found_implication impl = iter2.get_next();

        sat_status status = new_decision(impl.implication,
                                         conflicting_clause, implication_list);

        if (status == sat_status::UNSAT) {
            return sat_status::UNSAT;
        }

    }


    return sat_status::UNDEF;

}

__device__ bool WatchedClausesList::process_clause(WatchedClause *watched_clause,
        Decision decision,
        GPULinkedList<found_implication>& implication_list)
{

    Lit first_watched_literal = watched_clause->
                                clause.literals[watched_clause->watched_lit_index_1];

    Lit second_watched_literal = watched_clause->
                                 clause.literals[watched_clause->watched_lit_index_2];

#ifdef USE_ASSERTIONS
    assert(var(decision.literal) == var(first_watched_literal) ||
           var(decision.literal) == var(second_watched_literal));
#endif

    bool first_watched_consistent = is_currently_consistent(first_watched_literal, implication_list);
    bool second_watched_consistent = is_currently_consistent(second_watched_literal, implication_list);


    // First case, both consistent.
    if (first_watched_consistent && second_watched_consistent) {
        return false;
    }

    int pending_literals_switches = !first_watched_consistent && !second_watched_consistent
                                    ? 2 : 1;

    int first_other_lit_index = -1;
    int second_other_lit_index = -1;

    int index = 0;

    while (pending_literals_switches > 0 && index < watched_clause->clause.n_lits) {
        Lit literal = watched_clause->clause.literals[index];

        if (literal != first_watched_literal && literal != second_watched_literal) {
            //sat_status status = vars_handler->literal_status(literal);

            //if (status != sat_status::UNSAT)
            if (is_currently_consistent(literal, implication_list)) {
                if (first_other_lit_index == -1) {
                    first_other_lit_index = index;
                }
                else {
                    second_other_lit_index = index;
                }

                pending_literals_switches--;
            }
        }

        index++;
    }

    // second case, exactly one consistent
    if (first_watched_consistent || second_watched_consistent) {
        // If there is no other consistent var to watch, the clause is unit.
        if (first_other_lit_index == -1) {
            int index;
            if (first_watched_consistent) {
                index = watched_clause->watched_lit_index_1;
            }
            else {
                index = watched_clause->watched_lit_index_2;
            }
            handle_implication(watched_clause, watched_clause->clause.literals[index],
                               implication_list, decision);
            return false;
        }
        // If we do have another consistent var to watch, we simply watch it.
        else {
            int index;
            if (!first_watched_consistent) {
                index = watched_clause->watched_lit_index_1;
            }
            else {
                index = watched_clause->watched_lit_index_2;
            }

            structure.switch_watched_literal(index, first_other_lit_index, watched_clause);
            return false;
        }
    }

    // Third case, all are inconsistent! (no needs for 'ifs',
    //if it is here, all are inconsistent).

    if (first_other_lit_index != -1 && second_other_lit_index != -1) {
        // We can replace all inconsistent literals!
        structure.switch_watched_literal(watched_clause->watched_lit_index_1,
                                         first_other_lit_index, watched_clause);
        structure.switch_watched_literal(watched_clause->watched_lit_index_2,
                                         second_other_lit_index, watched_clause);
        return false;
    }

    if (first_other_lit_index != -1 || second_other_lit_index != -1) {
        int index;
        if (first_other_lit_index != -1) {
            index = first_other_lit_index;
        }
        else {
            index = second_other_lit_index;
        }

        // The clause is unit
        // TODO here comes the code to implicate.
        handle_implication(watched_clause, watched_clause->clause.literals[index],
                           implication_list, decision);
        return false;
    }

    // Last case, all watched are inconsistent as well as any other to replace them.
    // Now, we have a conflict.
    // TODO handle conflict here.

    handle_conflict(watched_clause);

    return true;


}

__device__ bool WatchedClausesList::is_currently_consistent(Lit literal,
        GPULinkedList<found_implication>& partial_implications)
{
    if (!vars_handler->is_consistent(literal)) {
        return false;
    }

    auto iter = partial_implications.get_iterator();

    while (iter.has_next()) {
        found_implication fi = iter.get_next();

        if (fi.implication.literal == ~literal) {
            return false;
        }
    }
    return true;

}

__device__ void WatchedClausesList::handle_backtrack()
{
    structure.unblock_all();
}

__device__ void WatchedClausesList::reset()
{
    structure.reset();
}

__device__ void WatchedClausesList::handle_implication(WatchedClause *watched_clause,
        Lit unit_literal,
        GPULinkedList<found_implication>& implication_list, Decision original_implication)
{

    if (literal_status(unit_literal, implication_list) == sat_status::UNDEF) {

        Decision d;
        d.decision_level = vars_handler->get_decision_level();
        d.literal = unit_literal;
        //d.implicated_from_formula = original_implication.implicated_from_formula;

        bool impl_from_formula = false;
        if (original_implication.implicated_from_formula) {
            Clause c = watched_clause->clause;
            impl_from_formula = vars_handler->should_implicate_from_formula(c, unit_literal);
        }
        d.implicated_from_formula = impl_from_formula;

        found_implication implication;
        implication.implication = d;
        implication.implicating_clause = &(watched_clause->clause);
        implication_list.push_back(implication);


    }

    structure.block_clause(watched_clause);

}

__device__ sat_status WatchedClausesList::literal_status(Lit literal,
        GPULinkedList<found_implication>& implication_list
                                                        )
{
    sat_status first_test = vars_handler->literal_status(literal);

    if (first_test != sat_status::UNDEF) {
        return first_test;
    }

    GPULinkedList<found_implication>::LinkedListIterator iter = implication_list.get_iterator();

    while (iter.has_next()) {
        found_implication fi = iter.get_next();

        if (fi.implication.literal == literal) {
            return sat_status::SAT;
        }
        if (fi.implication.literal == ~literal) {
            return sat_status::UNSAT;
        }

    }
    return sat_status::UNDEF;

}

__device__ void WatchedClausesList::handle_conflict(WatchedClause *watched_clause)
{
    structure.block_clause(watched_clause);
}

__device__ void WatchedClausesList::new_clause(const Clause *clause)
{
    Lit dummy;
    new_clause(clause, dummy, false);
}
__device__ void WatchedClausesList::new_clause(const Clause *clause, Lit& implicated, bool check_status)
{
    // TODO define best way to do this:
    //WatchedClause * watched_clause = (WatchedClause*) malloc(sizeof(WatchedClause));
    WatchedClause *watched_clause = repository->get_ptr(next_watched_clause_index);
    next_watched_clause_index++;
    structure.n_added_clauses++;
    new_clause(*clause, watched_clause, implicated, check_status);
}


__device__ void WatchedClausesList::new_clause(Clause clause,
        WatchedClause *watched_clause, Lit& lit, bool check_status)
{

#ifdef USE_ASSERTIONS
    assert(clause.n_lits > 1);
#endif

    watched_clause->clause = clause;

    if (check_status) {
        sat_status status = vars_handler->clause_status(clause, &lit);

        if (status == sat_status::UNSAT || lit.x != -1) {
            structure.block_clause(watched_clause);
        }
        else {
            structure.watch_clause(watched_clause, false);
        }

    }
    else {
        structure.watch_clause(watched_clause, false);
    }
}


__device__ void WatchedClausesList::add_all_clauses(CUDAClauseVec const& formula)
{
    for (int i = 0; i < formula.size_of(); i++) {
        const Clause *c = formula.get_ptr(i);
        if (c->n_lits > 1) {
            new_clause(c);
        }
    }
}


__device__ void WatchedClausesList::print_structure()
{
    structure.print_structure();
}

__device__ bool WatchedClausesList::check_consistency()
{
    return structure.check_consistency();
}

__device__ bool WatchedClausesList::check_clause(Clause *clause, bool& blocked)
{
    return structure.check_clause(clause, blocked);
}

__device__ bool WatchedClausesList::contains(Var v, Clause c)
{
    return structure.contains(v, c);
}
__device__ bool WatchedClausesList::blocked_contains(Clause c)
{
    return structure.blocked_contains(c);
}
__host__ __device__ void print_watch_clause(WatchedClause& wc)
{
    print_clause(wc.clause);
    printf(" (");
    print_lit(wc.clause.literals[wc.watched_lit_index_1]);
    printf(",");
    print_lit(wc.clause.literals[wc.watched_lit_index_2]);
    printf(")");
}
