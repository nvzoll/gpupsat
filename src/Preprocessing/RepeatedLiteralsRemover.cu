#include "RepeatedLiteralsRemover.cuh"

#include <algorithm>

RepeatedLiteralsRemover::RepeatedLiteralsRemover(
    std::vector<Clause>& f, int n)
    : formula { f }
    , n_vars { n }
    , stats(n_vars)
    , status { sat_status::UNDEF }
{

}

RepeatedLiteralsRemover::~RepeatedLiteralsRemover()
{

}

sat_status RepeatedLiteralsRemover::get_status()
{
    return status;
}

bool RepeatedLiteralsRemover::process_clause(Clause& clause)
{
    std::fill(std::begin(stats), std::end(stats), sat_status::UNDEF);

    for (size_t i = 0; i < clause.n_lits; i++) {
        Var v = var(clause.literals[i]);
        bool s = sign(clause.literals[i]);
        sat_status stat = s ? sat_status::SAT : sat_status::UNSAT;

        if (stats[v] == sat_status::UNDEF) {
            stats[v] = stat;
        }
        else {
            if (stats[v] == stat) {
                remove_literal(clause, i);
                i--;
            }
            else {
                return true;
            }
        }

    }
    return false;

}

void RepeatedLiteralsRemover::process()
{
    auto it = std::remove_if (std::begin(formula), std::end(formula),
        [this] (Clause& cl) -> bool { return process_clause(cl); });

    if (it != std::end(formula)) {
        formula.erase(it, std::end(formula));
    }

    status = (formula.size() == 0) ? sat_status::SAT : sat_status::UNDEF;
}
