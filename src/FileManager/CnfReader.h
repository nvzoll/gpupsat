#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <map>
#include <memory>

#include "FormulaData.cuh"

class CnfManager
{
    int max_var = -1;

    int header_n_clauses;
    int header_n_vars;

    int var_with_more_occurrences;
    int max_occurrence_of_var;

    Lit current_clause_lits[MAX_CLAUSE_SIZE];
    int current_clause_size = 0;

    FormulaData *formula_data;

    std::unique_ptr<std::map<int, int>> occurrences;

    void set_header(int vars, int clauses)
    {
        header_n_vars = vars;
        header_n_clauses = clauses;
    }

    void start_new_clause()
    {
        current_clause_size = 0;
    }

    void add_clause();
    void add_lit(Var v, bool sign);

public:
    bool read_cnf(const char *file, FormulaData& data);

    CnfManager() : occurrences { new std::map<int, int> } {}
};
