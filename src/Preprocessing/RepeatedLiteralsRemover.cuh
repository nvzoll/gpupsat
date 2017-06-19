#ifndef __REPEATEDLITERALSREMOVER_CUH__
#define __REPEATEDLITERALSREMOVER_CUH__

#include "SATSolver/SolverTypes.cuh"
#include <stdlib.h>
#include <vector>


class RepeatedLiteralsRemover
{
private:
    std::vector<Clause>& formula;
    int n_vars;
    sat_status status;

    // Auxiliary attributes
    std::vector<sat_status> stats;

    /**
     * Process clause, removing repeated literals.
     * Return true if the clause must be removed, false otherwise.
     */
    bool process_clause(Clause& clause);

public:
    RepeatedLiteralsRemover(std::vector<Clause>& formula, int n_vars);
    ~RepeatedLiteralsRemover();

    void process();
    sat_status get_status();

};

#endif /* __REPEATEDLITERALSREMOVER_CUH__ */
