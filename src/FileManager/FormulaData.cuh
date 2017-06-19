#ifndef __FORMULADATA_CUH__
#define __FORMULADATA_CUH__

#include <assert.h>
#include <vector>

#include "Utils/CUDAClauseVec.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "SATSolver/Configs.cuh"

class FormulaData
{

private:
    CUDAClauseVec formula_dev;
    std::vector<Clause> formula_host;
    std::vector<Lit> solved_literals;
    int n_clauses;
    int n_vars;
    int clauses_capacity;
    Var most_common_var;
    int frequency_of_most_common_var;
    int largest_clause_size;
    sat_status status_after_preprocessing;
    bool remove_unary;

public:
    FormulaData(int max_n_clauses, bool remove_unary);
    void add_clause(Lit *literals, int clause_size);
    void set_n_vars(int n_vars);
    void set_most_common_var(Var var, int frequency);

    CUDAClauseVec get_formula_dev();
    std::vector<Clause> *get_formula_host();
    int get_n_clauses();
    int get_n_vars();
    Var get_most_common_var();
    int get_frequency_of_most_common_var();
    int get_largest_clause_size();
    /**
     * Copies the clauses that are on the host to the device.
     * *** Make sure they have not been copied yet, otherwise they will be copied a second time.
     */
    void copy_host_clauses_to_dev();
    std::vector<Lit> const& get_solved_literals() const;
    sat_status get_status_after_preprocessing();

    __host__ void print_fomula();
};

#endif /* __FORMULADATA_CUH__ */
