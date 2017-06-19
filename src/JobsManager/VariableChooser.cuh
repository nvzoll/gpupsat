#ifndef __VARIABLECHOOSER_CUH__
#define __VARIABLECHOOSER_CUH__
#include <assert.h>
#include <cstdio>
#include <vector>

#include "SATSolver/SolverTypes.cuh"

struct Evaluation {
    Var var;
    int evaluation;
};

class VariableChooser
{
public:
    VariableChooser(std::vector<Clause> const& formula, size_t n_vars);
    ~VariableChooser();
    void evaluate();
    bool has_next_var();
    Var next_var();
private:
    std::vector<Clause> const& formula;
    std::vector<Evaluation> evaluations;
    std::vector<Evaluation>::iterator next_var_iter;
    size_t n_vars;
    bool evaluated;
};

#endif /* __VARIABLECHOOSER_CUH__ */
