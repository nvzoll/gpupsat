#include "FormulaData.cuh"
#include "Preprocessing/UnaryClausesRemover.cuh"
#include "Preprocessing/RepeatedLiteralsRemover.cuh"

FormulaData::FormulaData(int max_n_clauses, bool remove_unary)
    : formula_dev(max_n_clauses)
    , formula_host()
    , clauses_capacity { max_n_clauses }
    , n_clauses { 0 }
    , n_vars { 0 }
    , largest_clause_size { 0 }
    , status_after_preprocessing { sat_status::UNDEF }
    , remove_unary { remove_unary }
{

}

void FormulaData::add_clause(Lit *literals, int clause_size)
{
    assert(n_clauses < clauses_capacity);

    Clause c_host;

    create_clause_host(clause_size, c_host);

    for (int i = 0; i < clause_size; i++) {
        addLitToHost(literals[i], c_host);
    }

    if (clause_size > largest_clause_size) {
        largest_clause_size = clause_size;
    }

    formula_host.push_back(c_host);
    n_clauses++;

}

void FormulaData::set_n_vars(int n_vars)
{
    this->n_vars = n_vars;
}

void FormulaData::set_most_common_var(Var var, int frequency)
{
    this->most_common_var = var;
    this->frequency_of_most_common_var = frequency;
}

Var FormulaData::get_most_common_var()
{
    return most_common_var;
}
int FormulaData::get_frequency_of_most_common_var()
{
    return frequency_of_most_common_var;
}

CUDAClauseVec FormulaData::get_formula_dev()
{
    return formula_dev;
}
std::vector<Clause> *FormulaData::get_formula_host()
{
    return &formula_host;
}

int FormulaData::get_n_clauses()
{
    return n_clauses;
}
int FormulaData::get_n_vars()
{
    return n_vars;
}

int FormulaData::get_largest_clause_size()
{
    return largest_clause_size;
}

void FormulaData::copy_host_clauses_to_dev()
{
    status_after_preprocessing = sat_status::UNDEF;

    RepeatedLiteralsRemover rlr(formula_host, n_vars);
    rlr.process();
    status_after_preprocessing = rlr.get_status();

    if (remove_unary) {
        if (status_after_preprocessing == sat_status::UNDEF) {
            UnaryClausesRemover ucr(formula_host, n_vars);
            ucr.process();
            status_after_preprocessing = ucr.get_status();
            solved_literals = ucr.get_solved_literals();
        }
    }

    if (status_after_preprocessing == sat_status::UNDEF) {
        formula_dev.alloc_and_copy_to_dev(formula_host);
    }

    n_clauses = formula_dev.size_of();

}

sat_status FormulaData::get_status_after_preprocessing()
{
    return status_after_preprocessing;
}

std::vector<Lit> const& FormulaData::get_solved_literals() const
{
    return solved_literals;
}

__host__ void FormulaData::print_fomula()
{
    printf("Formula with %zu clauses:", formula_host.size());

    for (Clause const& cl : formula_host) {
        print_clause(cl);
        printf("\n");
    }

}
