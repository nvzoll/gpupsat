#include "WatchedClausesList.cuh"

__device__ WatchedClausesList::ClauseListStructure::ClauseListStructure(
    int n_vars,
    VariablesStateHandler *var_handler)
    : blocked_clauses()
    , vars_handler { var_handler }
    , n_vars { n_vars }
{
    watched_clauses_per_var = new GPULinkedList<WatchedClause *>[n_vars];

    GPULinkedList<WatchedClause *> list;
    for (int i = 0; i < n_vars; i++) {
        watched_clauses_per_var[i] = list;
    }

    n_added_clauses = 0;
}

__device__ WatchedClausesList::ClauseListStructure::ClauseListStructure(
    int n_vars,
    VariablesStateHandler *var_handler,
    watched_clause_node_t *node_repository)
    : blocked_clauses(node_repository)
    , vars_handler { var_handler }
    , n_vars { n_vars }
{
    watched_clauses_per_var = new GPULinkedList<WatchedClause *>[n_vars];

#ifdef USE_ASSERTIONS
    assert(watched_clauses_per_var != nullptr);
#endif

    GPULinkedList<WatchedClause *> list(node_repository);

    for (int i = 0; i < n_vars; i++) {
        watched_clauses_per_var[i] = list;
    }

    n_added_clauses = 0;
}

__device__ void WatchedClausesList::ClauseListStructure::add_clause(WatchedClause *clause, Var var)
{
#ifdef USE_ASSERTIONS
    assert(var < n_vars && var >= 0);
#endif

    watched_clauses_per_var[var].add_first(clause);

}
__device__ bool WatchedClausesList::ClauseListStructure::remove_clause(Clause *clause_ptr, Var var)
{
    GPULinkedList<WatchedClause *> *list = &watched_clauses_per_var[var];

    auto iter = list->get_iterator();
    while (iter.has_next()) {
        WatchedClause *wc = iter.get_next();
        if (wc->clause == *clause_ptr) {
            iter.remove();
            return true;
        }
    }

    return false;

}
__device__ WatchedClause *WatchedClausesList::ClauseListStructure::get_clause(Var var,
        int clause_index)
{
    return watched_clauses_per_var[var].get(clause_index);
}

__device__ int WatchedClausesList::ClauseListStructure::clauses_list_size(Var var)
{
#ifdef USE_ASSERTIONS
    assert(var < n_vars && var >= 0);
#endif
    return watched_clauses_per_var[var].size();
}

__device__ void WatchedClausesList::ClauseListStructure::switch_watched_literal(
    int old_watched_index,
    int new_watched_index,
    WatchedClause *clause)
{
    Var old_var = var(clause->clause.literals[old_watched_index]);
    Var new_var = var(clause->clause.literals[new_watched_index]);
#ifdef USE_ASSERTIONS
    assert(old_var >= 0 && old_var < n_vars &&
           new_var >= 0 && new_var < n_vars );
#endif

    if (clause->watched_lit_index_1 == old_watched_index) {
#ifdef USE_ASSERTIONS
        assert(clause->watched_lit_index_2 != new_watched_index);
#endif
        clause->watched_lit_index_1 = new_watched_index;
    }
    else {
#ifdef USE_ASSERTIONS
        assert(clause->watched_lit_index_2 == old_watched_index &&
               clause->watched_lit_index_1 != new_watched_index);
#endif
        clause->watched_lit_index_2 = new_watched_index;

    }

    remove_clause(&clause->clause, old_var);
    add_clause(clause, new_var);

}

__device__
GPULinkedList<WatchedClause *>::LinkedListIterator
WatchedClausesList::ClauseListStructure::get_iterator(Var var)
{
#ifdef USE_ASSERTIONS
    if (!(var >= 0 && var < n_vars)) {
        printf("Getting iterator for invalid var %d\n", var);
        print_structure();
        assert(false);
    }
#endif
    return watched_clauses_per_var[var].get_iterator();
}

__device__ void WatchedClausesList::ClauseListStructure::
watch_clause(WatchedClause *watched_clause, bool test_consistency)
{
    int smallest;
    int second_smallest;

    get_two_literals_with_fewer_clauses(&watched_clause->clause,
                                        smallest, second_smallest, test_consistency);

#ifdef USE_ASSERTIONS
    assert(smallest != second_smallest && smallest >= 0 && second_smallest >= 0 &&
           smallest < watched_clause->clause.n_lits &&
           second_smallest < watched_clause->clause.n_lits
          );
#endif

    watched_clause->watched_lit_index_1 = smallest;
    watched_clause->watched_lit_index_2 = second_smallest;

    Lit lit1 = watched_clause->clause.literals[watched_clause->watched_lit_index_1];
    Lit lit2 = watched_clause->clause.literals[watched_clause->watched_lit_index_2];

    add_clause(watched_clause, var(lit1));
    add_clause(watched_clause, var(lit2));
}

__device__ void WatchedClausesList::ClauseListStructure::
get_two_literals_with_fewer_clauses(Clause *c, int& smallest, int& second_smallest,
                                    bool test_consistency)
{
    int smallest_value = INT_MAX - 1;
    int second_smallest_value = INT_MAX;

    for (int i = 0; i < c->n_lits; i++) {
        if (!test_consistency || this->vars_handler->is_consistent(c->literals[i])) {
            Var v = var(c->literals[i]);
            int current_size = watched_clauses_per_var[v].size();

            if (current_size < smallest_value) {
                second_smallest = smallest;
                second_smallest_value = smallest_value;
                smallest_value = current_size;
                smallest = i;
            }
            else {
                if (current_size < second_smallest_value) {
                    second_smallest_value = current_size;
                    second_smallest = i;
                }
            }
        }
    }

#ifdef USE_ASSERTIONS
    assert(smallest_value > -1 && second_smallest_value > -1 &&
           smallest != second_smallest);
#endif
}

__device__ void WatchedClausesList::ClauseListStructure::block_clause(WatchedClause *watched_clause)
{

#ifdef USE_ASSERTIONS
    //if (blocked_clauses.contains(watched_clause))
    //{
    //printf("Watch clause:\n");
    //print_watch_clause(*watched_clause);
    //printf("\n");
    //print_structure();
    //assert(false);
    //}

    assert(!blocked_clauses.contains(watched_clause));
    assert(watched_clause->watched_lit_index_1 >= 0 &&
           watched_clause->watched_lit_index_1 < watched_clause->clause.n_lits);
    assert(watched_clause->watched_lit_index_2 >= 0 &&
           watched_clause->watched_lit_index_2 < watched_clause->clause.n_lits);
#endif

    Var v1 = var(watched_clause->clause.literals[watched_clause->watched_lit_index_1]);
    Var v2 = var(watched_clause->clause.literals[watched_clause->watched_lit_index_2]);

    remove_clause(&watched_clause->clause, v1);

    remove_clause(&watched_clause->clause, v2);

    blocked_clauses.add_first(watched_clause);

#ifdef USE_ASSERTIONS
    assert(!watched_clauses_per_var[v1].contains(watched_clause) &&
           !watched_clauses_per_var[v2].contains(watched_clause));
#endif

}

__device__ void WatchedClausesList::ClauseListStructure::
unblock_clause(WatchedClause *watched_clause)
{
#ifdef USE_ASSERTIONS
    assert(blocked_clauses.contains(watched_clause));
#endif

    watch_clause(watched_clause, true);

    bool removed = blocked_clauses.remove_obj(watched_clause);

#ifdef USE_ASSERTIONS
    assert(removed);
#endif
}

__device__ void WatchedClausesList::ClauseListStructure::unblock_all()
{
    auto iter =
        blocked_clauses.get_iterator();
    while (iter.has_next()) {
        WatchedClause *wc = iter.get_next();
        // TODO This can be improved.
        //if (vars_handler->n_consistent_literals(wc->clause) >= 2)
        if (vars_handler->is_unresolved(&wc->clause)) {
            watch_clause(wc, true);
            iter.remove();
        }
    }
}

__device__ bool WatchedClausesList::ClauseListStructure::purge_clause_from_blocked(Clause c)
{
    auto iter = blocked_clauses.get_iterator();

    while (iter.has_next()) {
        WatchedClause *wc = iter.get_next();
        if (wc->clause == c) {
            iter.remove();
            return true;
        }
    }

    return false;
}

__device__ void WatchedClausesList::ClauseListStructure::purge_clause(Clause c)
{
    if (purge_clause_from_blocked(c)) {
        return;
    }

    Var other_var = -1;

    bool removed = false;

    for (int i = 0; i < c.n_lits; i++) {
        Var v = var(c.literals[i]);

        auto iter = watched_clauses_per_var[v].get_iterator();
        while (iter.has_next()) {
            WatchedClause *wc = iter.get_next();

            if (wc->clause == c) {
                other_var = var(wc->clause.literals[wc->watched_lit_index_1]) == v ?
                            var(wc->clause.literals[wc->watched_lit_index_2]) :
                            var(wc->clause.literals[wc->watched_lit_index_1]);

                iter.remove();
                removed = true;
                break;
            }
        }
        if (removed) {
            break;
        }
    }

#ifdef USE_ASSERTIONS
    assert(removed && other_var != -1);
#endif

    auto iter = watched_clauses_per_var[other_var].get_iterator();
    while (iter.has_next()) {
        WatchedClause *wc = iter.get_next();
        if (wc->clause == c) {
            iter.remove();
            return;
        }
    }

#ifdef USE_ASSERTIONS
    assert(false);
#endif

}

__device__ void WatchedClausesList::ClauseListStructure::reset()
{
    auto iter = blocked_clauses.get_iterator();
    while (iter.has_next()) {
        WatchedClause *c = iter.get_next();
        watch_clause(c, false);
        iter.remove();
    }
}

__device__ bool WatchedClausesList::ClauseListStructure::check_consistency()
{

    int clauses_count = 0;
    for (int i = 0; i < n_vars; i++) {
        GPULinkedList<WatchedClause *> list = watched_clauses_per_var[i];

        auto iter = list.get_iterator();
        while (iter.has_next()) {
            WatchedClause *wc = iter.get_next();
            clauses_count++;
            if (wc->watched_lit_index_1 < 0 ||
                wc->watched_lit_index_1 >= wc->clause.n_lits) {
                printf("Invalid index %d for first watched literal in "
                       "clause with %d elements watching var %d.\n",
                       wc->watched_lit_index_1,
                       wc->clause.n_lits,
                       i
                      );
                return false;
            }
            if (wc->watched_lit_index_2 < 0 ||
                wc->watched_lit_index_2 >= wc->clause.n_lits) {
                printf("Invalid index %d for second watched literal in "
                       "clause with %d elements watching var %d.\n",
                       wc->watched_lit_index_2,
                       wc->clause.n_lits,
                       i
                      );
                return false;
            }
            if (wc->clause.literals[wc->watched_lit_index_1] ==
                wc->clause.literals[wc->watched_lit_index_2]
               ) {
                printf("Clause (");
                print_clause(wc->clause);
                printf(") is watching the same literal(");
                print_lit(wc->clause.literals[wc->watched_lit_index_1]);
                printf(") twice!\n");
                return false;
            }
            if ((!var(wc->clause.literals[wc->watched_lit_index_1]) == i) &&
                (!(var(wc->clause.literals[wc->watched_lit_index_2]) == i))
               ) {
                printf("Clause (");
                print_watch_clause(*(wc));
                printf(") does not watch literal of var %d, but it is in its list!\n", i);
                return false;
            }
            int other_var = var(wc->clause.literals[wc->watched_lit_index_1]) == i ?
                            var(wc->clause.literals[wc->watched_lit_index_2]) :
                            var(wc->clause.literals[wc->watched_lit_index_1]);
            if (!watched_clauses_per_var[other_var].contains(wc)) {
                printf("Clause(");
                print_watch_clause(*wc);
                printf(") found in list of var %d is not in the list of var %d\n", i, other_var);
                return false;
            }
            if (blocked_clauses.contains(wc)) {
                printf("Clause ");
                print_watch_clause(*wc);
                printf(" is both blocked and in the vars %d list!\n", i);
                return false;
            }
        }
        assert(list.check_consistency());

    }

    if (clauses_count % 2 != 0) {
        printf("The number of clauses found was not even and since two instances of each"
               "are stored, it should be!\n");
        return false;
    }

    if (((clauses_count / 2) + blocked_clauses.size()) != n_added_clauses) {
        printf("By counting (removing duplicates), %d clauses were found,"
               " while %d where added in the first place\n",
               (clauses_count / 2) + blocked_clauses.size(),
               n_added_clauses
              );

        return false;
    }

    auto iter2 = blocked_clauses.get_iterator();

    while (iter2.has_next()) {
        WatchedClause *wc = iter2.get_next();

        if (vars_handler->is_unresolved(&wc->clause)) {
            printf("Clause ");
            print_clause((wc->clause));
            printf(" is unresolved and is blocked! Unresolved clauses must not be blocked!\n");
            return false;
        }
    }

    return true;
}

__device__ bool WatchedClausesList::ClauseListStructure::check_clause(
    Clause *clause, bool& blocked)
{
    auto iter = blocked_clauses.get_iterator();

    while (iter.has_next()) {
        WatchedClause *wc = iter.get_next();

        if (wc->clause == *clause) {
            blocked = true;
            return true;
        }
    }

    blocked = false;

    for (int i = 0; i < clause->n_lits; i++) {
        Var v = var(clause->literals[i]);

        auto iter = watched_clauses_per_var[v].get_iterator();

        while (iter.has_next()) {
            WatchedClause *wc = iter.get_next();

            if (wc->clause == *clause) {
                return true;
            }
        }
    }

    return false;
}

__device__ bool WatchedClausesList::ClauseListStructure::contains(Var v, Clause c)
{
    auto iter = watched_clauses_per_var[v].get_iterator();

    while (iter.has_next()) {
        if (iter.get_next()->clause == c) {
            return true;
        }
    }
    return false;
}

__device__ bool WatchedClausesList::ClauseListStructure::blocked_contains(Clause c)
{
    auto iter = blocked_clauses.get_iterator();

    while (iter.has_next()) {
        if (iter.get_next()->clause == c) {
            return true;
        }
    }

    return false;
}

__device__ void WatchedClausesList::ClauseListStructure::print_structure()
{
    printf("Watched Variables List:\n");
    for (int i = 0; i < n_vars; i++) {
        printf("-> %d:\n", i);

        auto iter = watched_clauses_per_var[i].get_iterator();
        while (iter.has_next()) {
            printf("\t");
            print_watch_clause(*(iter.get_next()));
            printf("\n");
        }
    }

    printf("Blocked clauses:\n");
    auto iter = blocked_clauses.get_iterator();

    while (iter.has_next()) {
        printf("\t");
        print_watch_clause(*(iter.get_next()));
        printf("\n");
    }
}

__device__ void WatchedClausesList::ClauseListStructure::print_blocked_clauses()
{
    printf("Blocked clauses:\n");

    auto iter = blocked_clauses.get_iterator();
    while (iter.has_next()) {
        printf("\t");
        print_watch_clause(*(iter.get_next()));
        printf("\n");
    }
}
