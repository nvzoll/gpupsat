#include "UnaryClausesRemover.cuh"

#include <algorithm>

UnaryClausesRemover::UnaryClausesRemover(std::vector<Clause>& formula, int n_vars)
    : formula_host { formula }
    , status { sat_status::UNDEF }
    , n_vars { n_vars }
{

}

void UnaryClausesRemover::process_unary_clauses()
{
    auto it = std::remove_if(std::begin(formula_host), std::end(formula_host),
    [this] (Clause const & cl) -> bool {
        if (cl.n_lits == 1)
        {
            Lit lit = cl.literals[0];

            if (add(lit)) {
                return true;
            }
        }

        return false;
    });

    if (it != std::end(formula_host)) {
        formula_host.erase(it, std::end(formula_host));
    }

    /*
        if (solved_literals.size() == n_vars)
        {
            status = sat_status::SAT;
        }
    */
}

bool UnaryClausesRemover::add(Lit literal)
{

    for (Lit it : solved_literals) {
        if (literal == it) {
            return false;
        }

        if ((~literal) == it) {
            status = sat_status::UNSAT;
            return false;
        }

    }

    solved_literals.push_back(literal);
    return true;

}

sat_status UnaryClausesRemover::process_clause(
    std::vector<Lit>& implied_lits,
    Clause c,
    Lit& unit_lit)
{
    size_t unsat_lits = 0;

    for (size_t i = 0; i < c.n_lits; i++) {
        Lit current_lit = c.literals[i];
        bool current_unsat = false;

        for (Lit lit : implied_lits) {
            if (current_lit == lit) {
                unit_lit.x = -1;
                return sat_status::SAT;
            }

            if (current_lit == ~lit) {
                unsat_lits++;
                current_unsat = true;
                break;
            }
        }

        if (!current_unsat) {
            unit_lit = current_lit;
        }

    }

#ifdef USE_ASSERTIONS
    assert(unsat_lits >= 0 && unsat_lits <= c.n_lits);
#endif

    if (unsat_lits != c.n_lits - 1) {
        unit_lit.x = -1;
    }

    return unsat_lits == c.n_lits ? sat_status::UNSAT : sat_status::UNDEF;

}

void UnaryClausesRemover::clean_clause(std::vector<Lit>& implied_lits, Clause& c)
{
    for (size_t i = 0; i < c.n_lits; i++) {
        Lit clause_current = c.literals[i];
        for (Lit lit : implied_lits) {
#ifdef USE_ASSERTIONS
            assert(!(clause_current == lit));
#endif
            if (clause_current == ~lit) {
                remove_literal(c, i);
                i--;
            }
        }
    }
}

void UnaryClausesRemover::propagate_literals()
{
    auto it = formula_host.begin();

    while (it != formula_host.end()) {
        Clause cl = *it;
        Lit learnt;

        sat_status stat = process_clause(solved_literals, cl, learnt);

        if (learnt.x != -1) {
            add(learnt);
            if (status == sat_status::UNSAT) {
                return;
            }
        }

        if (stat == sat_status::SAT || learnt.x != -1) {
            it = formula_host.erase(it);
        }
        else {
            if (stat == sat_status::UNSAT) {
                status = sat_status::UNSAT;
                return;
            }
            else {
                clean_clause(solved_literals, *it);
                it++;
            }
        }

    }

    if (formula_host.size() == 0) {
        status = sat_status::SAT;
    }

}

void UnaryClausesRemover::process()
{
    process_unary_clauses();

    if (status != sat_status::UNDEF) {
        return;
    }

    int last_num_lit = 0;
    int current_num_lit = solved_literals.size();

    while (current_num_lit != last_num_lit) {
        propagate_literals();
        if (status != sat_status::UNDEF) {
            return;
        }

        last_num_lit = current_num_lit;
        current_num_lit = solved_literals.size();
    }

#ifdef USE_ASSERTIONS
    assert(test_results());
#endif
}

sat_status UnaryClausesRemover::get_status()
{
    return status;
}

bool UnaryClausesRemover::test_results()
{
    if (status != sat_status::UNDEF) {
        return true;
    }

    for (Clause const& cl : formula_host) {
        if (cl.n_lits <= 1) {
            printf("Empty or unary clause stored!\n");
            return false;
        }

        for (Lit lit : solved_literals) {
            for (size_t i = 0; i < cl.n_lits; i++) {
                if (var(lit) != var(cl.literals[i])) {
                    continue;
                }

                printf("The clause ");
                print_clause(cl);
                printf(" that contains the removed literal ");
                print_lit(lit);
                printf(" remains in the formula!\n");
                return false;
            }
        }
    }

    return true;
}

void UnaryClausesRemover::print_solved_literals()
{
    printf("Set literals = [ ");
    for (Lit it : solved_literals) {
        print_lit(it);
        printf(" ");
    }
    printf("]\n");
}

void UnaryClausesRemover::print_host_formula()
{
    printf("Formula:\n");
    for (Clause const& cl : formula_host) {
        print_clause(cl);
        printf("\n");
    }
}

std::vector<Lit> UnaryClausesRemover::get_solved_literals()
{
    // literals.insert(literals.end(), solved_literals.begin(), solved_literals.end());
    return solved_literals;
}
