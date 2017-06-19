#ifndef __UNARYCLAUSESREMOVER_CUH__
#define __UNARYCLAUSESREMOVER_CUH__

#include <vector>
#include "SATSolver/SolverTypes.cuh"
#include <assert.h>
#include "SATSolver/SATSolver.cuh"


class UnaryClausesRemover
{
private:
    std::vector<Clause>& formula_host;
    std::vector<Lit> solved_literals;
    sat_status status;
    int n_vars;

    bool add(Lit literal);
    sat_status process_clause(std::vector<Lit>& implied_lits, Clause c, Lit& unit_lit);
    void clean_clause(std::vector<Lit>& implied_lits, Clause& c);

public:
    UnaryClausesRemover(std::vector<Clause>& formula_host, int n_vars);
    void process_unary_clauses();
    void propagate_literals();
    void process();
    sat_status get_status();
    /**
     * Adds the current set literals to 'literals'
     */
    std::vector<Lit> get_solved_literals();

    // Test method
    void print_solved_literals();
    void print_host_formula();
    bool test_results();
};

#endif /* __UNARYCLAUSESREMOVER_CUH__ */
