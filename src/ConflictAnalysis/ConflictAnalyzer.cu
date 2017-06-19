#include "ConflictAnalyzer.cuh"

__device__ ConflictAnalyzer::ConflictAnalyzer(
    int n_var,
    const CUDAClauseVec *formula,
    VariablesStateHandler *handler,
    bool use_implication_graph,
    int max_implication_per_var,
    DecisionMaker *dec_maker,
    RuntimeStatistics *stats)

    : graph(use_implication_graph ? n_var : 1,
            use_implication_graph ? max_implication_per_var : 1)

    , backtracker(handler, nullptr, use_implication_graph ? (&graph) : nullptr, stats)

#ifdef USE_CLAUSE_LEARNING
    , learnt_clauses_manager { nullptr }
#endif
    , n_var { n_var }
    , formula { formula }
    , vars_handler { handler }
    , decision_maker { dec_maker }
    , use_implication_graph { use_implication_graph }
    , max_impl_per_var { max_implication_per_var }
    , stats { stats }
    , n_last_conflicts { 0 }
{
#ifdef USE_CLAUSE_LEARNING
    learnt_clauses_manager.set_watched_clauses(nullptr);
#endif
}

__device__
sat_status ConflictAnalyzer::handle_conflict(
    const Clause *c,
    Decision& implicated)
{
    n_last_conflicts++;

    if (use_implication_graph) {
        return handle_conflict_with_clause_learning(c, implicated);
    }
    else {
        sat_status stat =    handle_conflict_without_clause_learning();
        implicated = vars_handler->get_last_decision();
        return stat;
    }
}

/**
 * Propagates the currents decisions and implications, possibly generating more
 * implications (BCP through UP). Returns if the decisions have satisfied, unsatisfied or not
 * changed the current status of the formula.
 */

__device__ sat_status ConflictAnalyzer::handle_conflict_with_clause_learning(
    const Clause *c, Decision& implicated)
{

#ifdef DEBUG_THROUGH_VARIABLES_STATUS
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        //vars_handler->print_all();
    }
#endif

    if (vars_handler->get_decision_level() <= 0) {
        int decision_level_to_backtrack = -1;
        sat_status status = backtracker.handle_backtrack(
                                decision_level_to_backtrack);
        return sat_status::UNSAT;
    }

    add_conflict_to_graph(c);
    Decision next, highest;
    Clause learnt = learn_clause(next, highest);

    sat_status status = ConflictAnalyzer::
                        implicate_and_backtrack_from_learnt_clause(learnt,
                                highest, next, implicated);

    handle_learnt_clause(&learnt, highest, next);

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Conflict graph generated:\n");
        graph.print();
        printf("This clause can be learned:");
        print_clause(learnt);
        printf(", backtrack to: %d\n", next.decision_level);
        printf("\n");
        printf("The literal set in the highest decision level "
               "in this clause is:");
        print_lit(highest.literal);
        printf("\n");
        vars_handler->print_decisions();
    }
    return status;
}

__device__ sat_status ConflictAnalyzer::handle_conflict_without_clause_learning()
{
    return backtracker.handle_chronological_backtrack();
}

__device__ sat_status ConflictAnalyzer::
implicate_and_backtrack_from_learnt_clause(
    Clause const& learnt_clause, Decision highest,
    Decision second_highest, Decision& implicated)
{

    sat_status status;

    // Implicated from assumptions or from the implications of the formula?
    if (highest.decision_level == 0) {

        backtracker.handle_backtrack(0, false);
        return sat_status::UNSAT;
    }

    if (learnt_clause.n_lits == 1) {
        //printf("Unary clause:");
        //print_clause(learnt_clause);
        //printf("\n");
        status =  backtracker.handle_backtrack(1, false);
        implicated.decision_level = 0;
        implicated.implicated_from_formula = true;
        implicated.literal = learnt_clause.literals[0];

        make_decision(implicated);
        vars_handler->new_implication(implicated);

        return status;
    }

    if (second_highest.decision_level == highest.decision_level) {
        // This should not happen (asserting clause)
        assert(false);
        status =  backtracker.handle_backtrack(highest.decision_level);
        implicated = vars_handler->get_last_decision();
        implicated.implicated_from_formula = false;
    }
    else {
        status = backtracker.handle_backtrack(second_highest.decision_level + 1, false);
        implicated = highest;
        implicated.decision_level = vars_handler->get_decision_level();
        implicated.implicated_from_formula = false;
        add_implication(implicated, &learnt_clause);
    }

    return status;
}

__device__ void ConflictAnalyzer::handle_learnt_clause(
    const Clause *learnt, Decision highest, Decision second_highest)
{

    if (decision_maker != nullptr) {
        decision_maker->handle_learnt_clause(*learnt);
    }

#ifdef USE_CLAUSE_LEARNING
    if (learnt->n_lits < max_impl_per_var) {
        if (learnt->n_lits > 1) {
            learnt_clauses_manager.learn_clause(*learnt);
        }
        else {
            // TODO What if it is unary?
        }
    }
#endif
}

__device__ sat_status ConflictAnalyzer::propagate_all_clauses(Decision d,
        const Clause **conflicting_clause)
{
    int sat_clauses = 0;

    int last_n_implications = -10;
    while (last_n_implications != vars_handler->n_implications()) {
        sat_clauses = 0;
        last_n_implications = vars_handler->n_implications();

        for (int i = 0; i < formula->size_of(); i++) {
            const Clause *c = formula->get_ptr(i);
#ifdef USE_ASSERTIONS
            if (c->n_lits == 0) {
                printf("A formula clause is empty!\n");
                assert(false);
            }
#endif
            bool impl_formula = vars_handler->should_implicate_from_formula(*c, d.literal);
            sat_status status = clause_status(c, impl_formula);

            if (status == sat_status::UNSAT) {
                if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
                    printf("Clause ");
                    print_clause(*c);
                    printf(" of index %d is sat_status::UNSAT!\n", i);
                }

                *conflicting_clause = c;
                return sat_status::UNSAT;
            }


            if (status == sat_status::SAT) {
                sat_clauses++;

                if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
                    printf("Clause ");
                    print_clause(*c);
                    printf(" is sat_status::SAT!\n\n");
                }
            }
        }
    }

#ifdef USE_CLAUSE_LEARNING
    GPULinkedList<Clause> learnt_clauses;
    learnt_clauses_manager.copy_clauses(learnt_clauses);
    int learnt_size = learnt_clauses.size();

    GPULinkedList<Clause>::LinkedListIterator iterator = learnt_clauses.get_iterator();

    while (iterator.has_next()) {
        Clause *c = iterator.get_next_ptr();
#ifdef USE_ASSERTIONS
        if (c->n_lits == 0) {
            printf("A learnt clause is empty!\n");
            assert(false);
        }
#endif

        bool impl_formula = vars_handler->should_implicate_from_formula(*c, d.literal);
        sat_status status = clause_status(c, impl_formula);

        if (status == sat_status::UNSAT) {
            *conflicting_clause = c;
            return sat_status::UNSAT;
        }
        if (status == sat_status::SAT) {
            sat_clauses++;
        }

    }

    learnt_clauses.unalloc();

    return sat_clauses == formula->size_of() + learnt_size ? sat_status::SAT : sat_status::UNDEF;

#else // USE_CLAUSE_LEARNING

    return sat_clauses == formula->size_of() ? sat_status::SAT : sat_status::UNDEF;

#endif // USE_CLAUSE_LEARNING

}
__device__ sat_status ConflictAnalyzer::propagate(Decision d)
{
    n_last_conflicts = 0;

    make_decision(d);

#ifdef DEBUG_THROUGH_VARIABLES_STATUS
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("** Before propagating...\n");
        vars_handler->print_all();
    }
#endif

    d.implicated_from_formula = false;
    const Clause *conflicting_clause;
    sat_status status = propagate_all_clauses(d, &conflicting_clause);
    sat_status status_confl;

    while (status == sat_status::UNSAT) {
        Decision next;
        status_confl = handle_conflict(conflicting_clause, next);

        //vars_handler->print_all();
        //learnt_clauses_manager.print_learnt_clauses_repository();
        //return sat_status::UNSAT;

#ifdef USE_ASSERTIONS
        assert(next.decision_level > -1 || next.decision_level == NULL_DECISION_LEVEL);
        assert(graph.check_consistency());
        if (!vars_handler->check_consistency()) {
            vars_handler->print_all();
            assert(false);
        }
#endif


        if (status_confl != sat_status::UNDEF) {
            return status_confl;
        }

#ifdef DEBUG_THROUGH_VARIABLES_STATUS
        if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
            printf("** Before propagating...\n");
            vars_handler->print_all();
        }
#endif

        status = propagate_all_clauses(next, &conflicting_clause);
    }

    return status;
}

/**
 * This methods propagates the decisions, assumptions and implications.
 *
 * Requirement: decisions, assumptions and implications are disjoint (have not common literal)
 *
 */

__device__ inline sat_status ConflictAnalyzer::clause_status(const Clause *c, bool implicated_from_formula)
{

    Lit learnt;
    sat_status stat = vars_handler->clause_status(*c, &learnt);

    if (stat == sat_status::UNDEF && learnt.x != -1) {
        Decision implicated;
        implicated.literal = learnt;
        implicated.decision_level = vars_handler->get_decision_level();
        implicated.implicated_from_formula = implicated_from_formula;
        add_implication(implicated, c);
    }

    return stat;
}

__device__ sat_status ConflictAnalyzer::process_unit_clauses()
{
    for (int i = 0; i < formula->size_of(); i++) {
        const Clause c = formula->get(i);
        if (c.n_lits == 1) {
            Decision implied;
            implied.decision_level = 0;
            implied.implicated_from_formula = true;
            implied.literal = c.literals[0];
            vars_handler->new_implication(implied);
            make_decision(implied);
            sat_status status = propagate(implied);

            if (status != sat_status::UNDEF) {
                for (int i = 0;
                     i < vars_handler->n_implications(); i++) {
                    vars_handler->get_implication(i)->implicated_from_formula = true;
                }
                return status;
            }

        }
    }

    for (int i = 0;
         i < vars_handler->n_implications(); i++) {
        vars_handler->get_implication(i)->implicated_from_formula = true;
    }

    return sat_status::UNDEF;

}

__device__ sat_status ConflictAnalyzer::set_assumptions(
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> *assumptions
#else
    GPUStaticVec<Lit> *assumptions
#endif
)
{


    if (use_implication_graph) {
        for (int i = 0; i < vars_handler->n_assumptions(); i++) {
            Lit lit_assumption = vars_handler->get_assumption(i);
            Decision d;
            d.literal = lit_assumption;
            d.decision_level = 0;
            d.implicated_from_formula = false;
            sat_status assumptions_status = vars_handler->literal_status(lit_assumption, false);

            if (assumptions_status != sat_status::UNDEF) {
#ifdef USE_ASSERTIONS
                if (!graph.contains(var(lit_assumption))) {
                    printf("Assumptions ");
                    print_lit(lit_assumption);
                    printf(" is not sat_status::UNDEF, but it is not in the implication graph.\n");
                    vars_handler->print_all();

                    assert(false);
                }
#endif
                if (assumptions_status == sat_status::UNSAT) {
                    return sat_status::UNSAT;
                }
                if (assumptions_status == sat_status::SAT) {
                    continue;
                }
            }


            add_decision_to_graph(d);
        }

    }

    const Clause *dummy;
    Decision dummy_dec;
    dummy_dec.implicated_from_formula = false;
    sat_status status = ConflictAnalyzer::propagate_all_clauses(dummy_dec, &dummy);

#ifdef USE_ASSERTIONS
    assert(vars_handler->check_consistency());
#endif

    return status;
    //return sat_status::UNDEF;
}

/**
 * Must be call after removing assumptions, decisions and implications.
 */
__device__ void ConflictAnalyzer::reset()
{
    graph.reset();
}

__device__ void ConflictAnalyzer::add_decision_to_graph(Decision decision)
{

#ifdef USE_ASSERTIONS
    if (graph.contains(var(decision.literal))) {
        printf("Attempting to add ");
        print_decision(decision);
        printf(" to graph, which already contains the var.\n");
        print_graph();
        assert(false);
    }
#endif

    graph.set(decision);
}

__device__ void ConflictAnalyzer::add_implication(Decision implication,
        const Clause *clause)
{
    if (use_implication_graph) {
#ifdef USE_ASSERTIONS
        assert(!graph.contains(var(implication.literal)));
        bool assert_lit_in_clause = false;
#endif
        graph.set(implication);

        for (int i = 0; i < clause->n_lits; i++) {
            if (clause->literals[i] != implication.literal) {
#ifdef USE_ASSERTIONS
                assert(var(clause->literals[i]) != var(implication.literal));
#endif
                graph.link(var(clause->literals[i]), var(implication.literal), clause);
            }
#ifdef USE_ASSERTIONS
            else {
                assert_lit_in_clause = true;
            }
#endif
        }

#ifdef USE_ASSERTIONS
        // The clause MUST contain the learnt literal.
        assert(assert_lit_in_clause);
#endif
    }

    vars_handler->new_implication(implication);
}

__device__ bool ConflictAnalyzer::is_set_in_graph(Var var)
{
    return graph.is_set(var);
}

__device__ bool ConflictAnalyzer::check_consistency()
{
    return graph.check_consistency() && check_solver_consistency();
}

__device__ bool ConflictAnalyzer::check_solver_consistency()
{
    if (vars_handler->get_decision_level() < -1) {
        printf("Decision level (%d) is invalid\n", vars_handler->get_decision_level());
        return false;
    }
    for (int i = 0; i < vars_handler->n_decisions(); i++) {
        if (vars_handler->get_decision(i).decision_level < 0
            || vars_handler->get_decision(i).decision_level > vars_handler->get_decision_level()) {
            printf("Decision level (%d) of decision ");
            print_lit(vars_handler->get_decision(i).literal);
            printf(" is invalid!\n");
            return false;
        }
        if (vars_handler->is_var_free(var(vars_handler->get_decision(i).literal))) {
            printf("Decision ");
            print_lit(vars_handler->get_decision(i).literal);
            printf(" is on free vars list!\n");
            return false;
        }
    }
    for (int i = 0; i < vars_handler->n_implications(); i++) {
        if (vars_handler->get_implication(i)->decision_level < 0
            || vars_handler->get_implication(i)->decision_level > vars_handler->get_decision_level()) {
            printf("Decision level (%d) of implication ");
            print_lit(vars_handler->get_implication(i)->literal);
            printf(" is invalid!\n");
            return false;
        }
        if (vars_handler->is_var_free(var(vars_handler->get_implication(i)->literal))) {
            printf("Implication ");
            print_lit(vars_handler->get_implication(i)->literal);
            printf(" is on free vars list!\n");
            return false;
        }
    }
    for (int i = 0; i < vars_handler->n_assumptions(); i++) {
        if (vars_handler->is_var_free(var(vars_handler->get_assumption(i)))) {
            printf("Assumption ");
            print_lit(vars_handler->get_assumption(i));
            printf(" is on free vars list!\n");
            return false;
        }
    }

    return true;
}
__device__ void ConflictAnalyzer::print_graph()
{
    printf("Graph:\n");
    graph.print();
}

__device__ void ConflictAnalyzer::make_decision(Decision decision)
{
    if (use_implication_graph) {
        add_decision_to_graph(decision);
    }

}

__device__ void ConflictAnalyzer::unmake_implication(Decision implication)
{
    vars_handler->free_from_implications(implication);
}


__device__ void ConflictAnalyzer::add_conflict_to_graph(const Clause *conflicting_clause)
{

#ifdef USE_ASSERTIONS
    assert(!graph.contains(graph.get_conflict_vertex_index()));
    if (!(conflicting_clause->n_lits > 0)) {
        printf("An empty conflicting clause was used.\n");
        formula->print_all();
        assert(false);
    }
#endif

    for (int i = 0; i < conflicting_clause->n_lits; i++) {
        graph.link_with_conflict(var(conflicting_clause->literals[i]),
                                 conflicting_clause, vars_handler->get_decision_level());
    }

#ifdef USE_ASSERTIONS
    int conflict_vertex_index = graph.get_conflict_vertex_index();
    if (!graph.is_set(conflict_vertex_index)) {
        printf("Just added conflict to graph (vertex %d), "
               "but conflict vertex is not set!\n",
               conflict_vertex_index);
        graph.print_vertex(conflict_vertex_index);
        assert(false);
    }
#endif

}

__device__ Clause ConflictAnalyzer::learn_clause(Decision& next_dec_level, Decision& highest)
{
    Clause clause = analyze_graph(graph, next_dec_level, highest);

#ifdef USE_ASSERTIONS
    for (int i = 0; i < clause.n_lits; i++) {
        if (vars_handler->literal_status(clause.literals[i]) == sat_status::UNDEF) {
            printf("Learnt clause ");
            print_clause(clause);
            printf(" should have all its literals set, but literal ");
            print_lit(clause.literals[i]);
            printf(" is free.\n");
            vars_handler->print_all();
            assert(false);
        }
    }

#endif

    return clause;
}
__device__ void ConflictAnalyzer::restart()
{
    backtracker.handle_backtrack(0, false);
}

__device__ int ConflictAnalyzer::get_n_last_conflicts()
{
    return n_last_conflicts;
}

__device__ bool ConflictAnalyzer::get_use_implication_graph()
{
    return use_implication_graph;
}
