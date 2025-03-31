#include "VariablesStateHandler.cuh"
#include <stddef.h>

__device__ VariablesStateHandler::VariablesStateHandler(
    int n_vars,
    const Var *dead_vars_elements_ptr, // Use raw pointer
    size_t dead_vars_size,             // Use size
    DecisionMaker *dec_maker)
    : free_vars(n_vars), decisions(n_vars), implications(n_vars), decision_maker{dec_maker}, n_vars{n_vars}
{
    vars_status = new sat_status[n_vars];

    for (int i = 0; i < n_vars; i++)
    {
        Var v = i;
        // Manually check if v is in the dead_vars array
        bool is_dead = false;
        if (dead_vars_elements_ptr != nullptr)
        { // Check if pointer is valid
            for (size_t j = 0; j < dead_vars_size; ++j)
            {
                if (dead_vars_elements_ptr[j] == v)
                {
                    is_dead = true;
                    break;
                }
            }
        }

        if (!is_dead)
        {
            free_vars.add(v);
        }
        else
        {
            if (dec_maker != nullptr)
            {
                dec_maker->block_var(v);
            }
        }
        vars_status[i] = sat_status::UNDEF;
    }

    assumptions = nullptr;
}

__device__ void VariablesStateHandler::set_assumptions(
#ifdef ASSUMPTIONS_USE_DYNAMICALLY_ALLOCATED_VECTOR
    GPUVec<Lit> *assumptions
#else
    GPUStaticVec<Lit> *assumptions
#endif
)
{
    this->assumptions = assumptions;

    if (assumptions == nullptr)
    {
        return;
    }

    // TODO improve this
    for (int i = 0; i < n_assumptions(); i++)
    {
        Var v = var(get_assumption(i));

        bool removed = free_vars.remove_obj(v);
        block_var(v);

        vars_status[v] = sign(get_assumption(i)) ? sat_status::SAT : sat_status::UNSAT;
    }
}

/**
 * TODO improve this.
 */
__device__ Decision VariablesStateHandler::get_last_decision()
{
    if (decisions.empty())
    {
        Decision d;
        d.decision_level = NULL_DECISION_LEVEL;
        return d;
    }

    return decisions.get(decisions.size_of() - 1);
}
__device__ size_t VariablesStateHandler::n_decisions()
{
    return decisions.size_of();
}
__device__ Decision VariablesStateHandler::get_decision(size_t index)
{
#ifdef USE_ASSERTIONS
    assert(index >= 0 && index < decisions.size_of());
#endif
    return decisions.get(index);
}
__device__ size_t VariablesStateHandler::n_implications()
{
    return implications.size_of();
}
__device__ Decision *VariablesStateHandler::get_implication(size_t index)
{
#ifdef USE_ASSERTIONS
    assert(index >= 0 && index < implications.size_of());
#endif
    return implications.get_ptr(index);
}
__device__ size_t VariablesStateHandler::n_assumptions()
{
    return assumptions->size_of();
}
__device__ Lit VariablesStateHandler::get_assumption(size_t index)
{
#ifdef USE_ASSERTIONS
    assert(index >= 0 && index < assumptions->size_of());
#endif
    return assumptions->get(index);
}

__device__ bool VariablesStateHandler::assumptions_set()
{
    return assumptions != nullptr;
}

__device__ int VariablesStateHandler::get_decision_level()
{
    return decision_level;
}
__device__ void VariablesStateHandler::set_decision_level(int decision_level)
{
    this->decision_level = decision_level;
}

__device__ sat_status VariablesStateHandler::literal_status(Lit lit)
{
    return literal_status(lit, true);
}

__device__ sat_status VariablesStateHandler::literal_status(Lit lit, bool check_assumptions)
{
#ifdef USE_ASSERTIONS
    assert(assumptions != nullptr);
#endif

    if (check_assumptions)
    {
        sat_status stat = vars_status[var(lit)];
        if (stat == sat_status::UNDEF)
        {
            return sat_status::UNDEF;
        }

        sat_status stat2 = sign(lit) ? sat_status::SAT : sat_status::UNSAT;

        if (stat == stat2)
        {
            return sat_status::SAT;
        }
        else
        {
            return sat_status::UNSAT;
        }
    }

    bool currently_satisfied = false;

    for (int j = 0; j < decisions.size_of(); j++)
    {
        if (decisions.get(j).literal == lit)
        {
            currently_satisfied = true;
        }

        if (decisions.get(j).literal == ~lit)
        {
            return sat_status::UNSAT;
        }
    }
    for (int j = 0; j < implications.size_of(); j++)
    {
        if (implications.get(j).literal == lit)
        {
            currently_satisfied = true;
        }

        if (implications.get(j).literal == ~lit)
        {
            return sat_status::UNSAT;
        }
    }

    if (check_assumptions)
    {
        for (int j = 0; j < assumptions->size_of(); j++)
        {
            if (assumptions->get(j) == lit)
            {
                currently_satisfied = true;
            }

            if (assumptions->get(j) == ~lit)
            {
                return sat_status::UNSAT;
            }
        }
    }

    if (currently_satisfied)
    {
        return sat_status::SAT;
    }
    else
    {
        return sat_status::UNDEF;
    }
}

__device__ sat_status VariablesStateHandler::clause_status(Clause clause,
                                                           Lit *learnt)
{
    int unsat_lits = 0;
    for (int i = 0; i < clause.n_lits; i++)
    {
        sat_status lit_stat = literal_status(clause.literals[i]);
        if (lit_stat == sat_status::SAT)
        {
            learnt->x = -1;
            return sat_status::SAT;
        }

        if (lit_stat == sat_status::UNSAT)
        {
            unsat_lits++;
        }

        if (lit_stat == sat_status::UNDEF)
        {
            *learnt = clause.literals[i];
        }
    }

    if (unsat_lits != (clause.n_lits - 1))
    {
        learnt->x = -1;
    }

    return unsat_lits == clause.n_lits ? sat_status::UNSAT : sat_status::UNDEF;
}

__device__ void VariablesStateHandler::increment_decision_level()
{
    decision_level++;
}

__device__ void VariablesStateHandler::reset()
{
    decision_level = 0;

    for (int i = 0; i < decisions.size_of(); i++)
    {
        free_vars.add(var(decisions.get(i).literal));
        free_var(var(decisions.get(i).literal));
    }
    decisions.clear();

    /*
    for (int i = 0; i < implications.size_of(); i++)
    {
        free_vars.add(var(implications.get(i).literal));
    }
    implications.clear();
    */

    for (int i = implications.size_of() - 1; i >= 0; i--)
    {
        Decision impl = implications.get(i);
        if (!impl.implicated_from_formula)
        {
            free_vars.add(var(impl.literal));
            free_var(var(impl.literal));
            implications.remove(i);
        }
    }

    for (int i = 0; i < assumptions->size_of(); i++)
    {
        free_vars.add(var(assumptions->get(i)));
        free_var(var(assumptions->get(i)));
    }
    // assumptions = nullptr;

    for (int i = 0; i < implications.size_of(); i++)
    {
        free_vars.remove_obj(var(implications.get(i).literal));
    }

    // TODO improve this
    for (int i = 0; i < n_vars; i++)
    {
        if (free_vars.contains(i))
        {
            vars_status[i] = sat_status::UNDEF;
        }
    }
}
__device__ bool VariablesStateHandler::is_consistent(Lit lit)
{
    sat_status stat = vars_status[var(lit)];
    sat_status stat2 = sign(lit) ? sat_status::SAT : sat_status::UNSAT;

    if (stat == sat_status::UNDEF || stat == stat2)
    {
        return true;
    }
    else
    {
        return false;
    }
}

__device__ int VariablesStateHandler::get_decision_level_of(Var v)
{
    for (int i = 0; i < decisions.size_of(); i++)
    {
        if (var(decisions.get(i).literal) == v)
        {
            return decisions.get(i).decision_level;
        }
    }

    for (int i = 0; i < implications.size_of(); i++)
    {
        if (var(implications.get(i).literal) == v)
        {
            return implications.get(i).decision_level;
        }
    }

    for (int i = 0; i < assumptions->size_of(); i++)
    {
        if (var(assumptions->get(i)) == v)
        {
            return 0;
        }
    }

    return NULL_DECISION_LEVEL;
}

__device__ int VariablesStateHandler::n_consistent_literals(Clause *clause)
{
    int number = 0;

    for (int i = 0; i < clause->n_lits; i++)
    {
        if (is_consistent(clause->literals[i]))
        {
            number++;
        }
    }
    return number;
}

__device__ void VariablesStateHandler::new_implication(Decision implication)
{

#ifdef USE_ASSERTIONS
    assert(free_vars.contains(var(implication.literal)));
#endif
    implications.add(implication);
    free_vars.remove_obj(var(implication.literal));
    block_var(var(implication.literal));
    vars_status[var(implication.literal)] = sign(implication.literal) ? sat_status::SAT : sat_status::UNSAT;
}

__device__ void VariablesStateHandler::add_many_implications(
    GPULinkedList<found_implication> &list_of_implications)
{

    GPULinkedList<found_implication>::LinkedListIterator iter = list_of_implications.get_iterator();

    while (iter.has_next())
    {
        new_implication(iter.get_next().implication);
    }
}

__device__ void VariablesStateHandler::new_decision(Decision decision)
{
#ifdef USE_ASSERTIONS
    assert(free_vars.contains(var(decision.literal)));
#endif
    decisions.add(decision);
    free_vars.remove_obj(var(decision.literal));
    block_var(var(decision.literal));
    vars_status[var(decision.literal)] = sign(decision.literal) ? sat_status::SAT : sat_status::UNSAT;
}

__device__ void VariablesStateHandler::undo_decision_or_implication(int index,
                                                                    GPUVec<Decision> &list)
{
    Decision dec = list.get(index);

    list.remove(index);

    free_vars.add(var(dec.literal));
    free_var(var(dec.literal));
    vars_status[var(dec.literal)] = sat_status::UNDEF;
}

__device__ void VariablesStateHandler::undo_decision(int index)
{
    undo_decision_or_implication(index, decisions);
}
__device__ void VariablesStateHandler::undo_implication(int index)
{
    undo_decision_or_implication(index, implications);
}

__device__ Decision VariablesStateHandler::backtrack_to(int new_decision_level)
{
    Decision next_decision;
    set_decision_level(new_decision_level);

    int i = 0;
    while (i < n_implications())
    {
        Decision *implic = get_implication(i);
        if (implic->decision_level > get_decision_level())
        {

            undo_implication(i);
            i--;
        }
        i++;
    }

    i = 0;
    while (i < decisions.size_of())
    {
        Decision dec = decisions.get(i);
        if (dec.decision_level > get_decision_level())
        {
            if (dec.decision_level == get_decision_level() + 1)
            {
                next_decision = dec;
            }

            undo_decision(i);
            i--;
        }
        i++;
    }

    return next_decision;
}
/**
 * Removes a decision, adding it back to free vars.
 * index_in_decisions: the index of the decision to be removed in the list of decisions.
 * (Must be greater or equal to 0 and smaller than the number of decisions)
 * Return: The removed decision.
 */
__device__ Decision VariablesStateHandler::unmake_decision(int index_in_decisions)
{
#ifdef USE_ASSERTIONS
    assert(index_in_decisions < decisions.size_of());
#endif
    Decision d = decisions.get(index_in_decisions);
    // decisions.remove(index_in_decisions);
    // free_vars.add(var(d.literal));
    undo_decision(index_in_decisions);
    return d;
}

__device__ bool VariablesStateHandler::is_unresolved(Clause *clause)
{
    if (clause->n_lits == 1)
    {
        return false;
    }
    int n_unsat = 0;

    for (int i = 0; i < clause->n_lits; i++)
    {
        sat_status stat = literal_status(clause->literals[i]);
        if (stat == sat_status::SAT)
        {
            return false;
        }
        if (stat == sat_status::UNSAT)
        {
            n_unsat++;
        }
    }

    if (n_unsat >= (clause->n_lits) - 1)
    {
        return false;
    }
    else
    {
        return true;
    }
}

__device__ Decision VariablesStateHandler::last_non_branched_decision()
{
    return last_non_branched_decision(decision_level + 1);
}

__device__ Decision VariablesStateHandler::last_non_branched_decision(
    int limit_decision_level)
{
    for (int i = decisions.size_of() - 1; i >= 0; i--)
    {
        Decision d = decisions.get(i);
        if (!d.branched && d.decision_level <= limit_decision_level)
        {
            return d;
        }
    }

    Decision d;
    d.decision_level = NULL_DECISION_LEVEL;
    return d;
}

__device__ bool VariablesStateHandler::is_var_free(Var v)
{
    return free_vars.contains(v);
}

__device__ void VariablesStateHandler::free_from_decisions(Decision decision)
{
#ifdef USE_ASSERTIONS
    assert(decisions.contains(decision) && !(free_vars.contains(var(decision.literal))));
#endif

    bool removed = decisions.remove_obj(decision);

#ifdef USE_ASSERTIONS
    assert(removed);
#endif

    free_vars.add(var(decision.literal));
    free_var(var(decision.literal));
}
__device__ void VariablesStateHandler::free_from_implications(Decision implication)
{
#ifdef USE_ASSERTIONS
    assert(implications.contains(implication) && !(free_vars.contains(var(implication.literal))));
#endif

    bool removed = implications.remove_obj(implication);

#ifdef USE_ASSERTIONS
    assert(removed);
#endif

    free_vars.add(var(implication.literal));
    free_var(var(implication.literal));
}
__device__ bool VariablesStateHandler::no_free_vars()
{
    return free_vars.empty();
}

__device__ bool VariablesStateHandler::no_decisions()
{
    return decisions.empty();
}
__device__ bool VariablesStateHandler::no_implications()
{
    return implications.empty();
}
__device__ bool VariablesStateHandler::no_assumptions()
{
    return assumptions->empty();
}

__device__ Var VariablesStateHandler::last_free_var()
{
    return free_vars.get(free_vars.size_of() - 1);
}
__device__ Var VariablesStateHandler::first_free_var()
{
    return free_vars.get(0);
}

__device__ void VariablesStateHandler::free_var(Var v)
{
    if (decision_maker != nullptr)
    {
        decision_maker->free_var(v);
    }
}
__device__ void VariablesStateHandler::block_var(Var v)
{
    if (decision_maker != nullptr)
    {
        decision_maker->block_var(v);
    }
}

__device__ bool VariablesStateHandler::is_set_as_implicated_from_formula(Var v)
{
    for (int i = 0; i < implications.size_of(); i++)
    {
        if (var(implications.get(i).literal) == v)
        {
            Decision implication = implications.get(i);
            return implication.implicated_from_formula;
        }
    }
    return false;
}
__device__ bool VariablesStateHandler::should_implicate_from_formula(
    Clause const &implication_clause, Lit implicated_literal)
{
    int count = 0;
    for (int i = 0; i < implication_clause.n_lits; i++)
    {
        if (var(implication_clause.literals[i]) == var(implicated_literal) ||
            is_set_as_implicated_from_formula(
                var(implication_clause.literals[i])))
        {
            count++;
        }
    }
    if (count == implication_clause.n_lits)
    {
        return true;
    }

    return false;
}

__device__ void VariablesStateHandler::print_decisions()
{
    printf("The current decisions are: ");

    for (int i = 0; i < decisions.size_of(); i++)
    {
        print_decision(decisions.get(i));
        printf(" ");
    }
    printf(" - decision level = %d\n", decision_level);
}

__device__ void VariablesStateHandler::print_free_vars()
{
    printf("Free vars: ");
    for (int i = 0; i < free_vars.size_of(); i++)
    {
        printf(" %d ", free_vars.get(i));
    }
    printf("\n");
}

__device__ void VariablesStateHandler::print_implications()
{
    printf("Implications: ");
    for (int i = 0; i < implications.size_of(); i++)
    {
        print_decision(implications.get(i));
        printf(" ");
    }
    printf("\n");
}

__device__ void VariablesStateHandler::print_assumptions()
{
    printf("Assumptions: ");
    if (assumptions != nullptr)
    {
        for (int i = 0; i < assumptions->size_of(); i++)
        {
            print_lit(assumptions->get(i));
            printf(" ");
        }
    }
    else
    {
        printf("nullptr");
    }
    printf("\n");
}

__device__ void VariablesStateHandler::print_var_status()
{

    printf("Status: ");
    for (int i = 0; i < n_vars; i++)
    {
        printf("%d=", i);
        print_status(vars_status[i]);
        printf(" ");
    }
    printf("\n");
}

__device__ void VariablesStateHandler::print_all()
{

    print_free_vars();
    print_assumptions();
    print_decisions();
    print_implications();
    print_var_status();
}

__device__ bool VariablesStateHandler::check_consistency()
{
    for (int i = 0; i < n_vars; i++)
    {
        int occurrences = 0;
        sat_status expected_status = sat_status::UNDEF;

        if (free_vars.contains(i))
        {
            occurrences++;
            expected_status = sat_status::UNDEF;
        }

        for (int j = 0; j < decisions.size_of(); j++)
        {
            Decision d = decisions.get(j);
            if (var(d.literal) == i)
            {
                occurrences++;
                expected_status = sign(d.literal) ? sat_status::SAT : sat_status::UNSAT;
            }
        }

        for (int j = 0; j < implications.size_of(); j++)
        {
            Decision d = implications.get(j);
            if (var(d.literal) == i)
            {
                occurrences++;
                expected_status = sign(d.literal) ? sat_status::SAT : sat_status::UNSAT;
            }
        }

        if (assumptions != nullptr)
        {
            for (int j = 0; j < assumptions->size_of(); j++)
            {
                Lit l = assumptions->get(j);
                if (var(l) == i)
                {
                    occurrences++;
                    expected_status = sign(l) ? sat_status::SAT : sat_status::UNSAT;
                }
            }
        }

        if (occurrences > 1)
        {
            printf("Variable %d was found in %d lists.\n", i, occurrences);
            return false;
        }
        if (expected_status != vars_status[i])
        {
            printf("The status ");
            print_status(expected_status);
            printf(" was found for var %d, but the one stored in 'vars_status' is ", i);
            print_status(vars_status[i]);
            printf(" and does not match.\n");
            return false;
        }
    }
    return true;
}
