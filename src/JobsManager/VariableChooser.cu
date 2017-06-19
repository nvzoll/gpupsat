#include "VariableChooser.cuh"
#include <math.h>
#include <algorithm>

VariableChooser::VariableChooser(std::vector<Clause> const& f, size_t n_vars)
    : formula { f }
    , evaluations(n_vars)
    , n_vars { n_vars }
    , evaluated { false }
{
    for (size_t i = 0; i < n_vars; i++) {
        Var v = i;
        evaluations[i].var = v;
        evaluations[i].evaluation = 0;
    }
}

VariableChooser::~VariableChooser()
{

}

void VariableChooser::evaluate()
{
    for (Clause const& c : formula) {
        for (size_t j = 0; j < c.n_lits; j++) {
            Var v = var(c.literals[j]);
            int index = v;

            evaluations[index].evaluation += v;
        }
    }

    std::sort(std::begin(evaluations), std::end(evaluations),
        [] (Evaluation const& l, Evaluation const& r) { return l.evaluation > r.evaluation; });

    next_var_iter = std::begin(evaluations);

    evaluated = true;
}

Var VariableChooser::next_var()
{
#ifdef USE_ASSERTIONS
    assert(evaluated && next_var_index >= 0);
#endif

    Var v = next_var_iter->var;
    next_var_iter++;

    return v;
}

bool VariableChooser::has_next_var()
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif

    return next_var_iter != std::end(evaluations);
}
