#include "CnfReader.h"

#include <cstdlib>

#include <boost/spirit/include/phoenix.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/qi_match.hpp>

using namespace boost::spirit;
using boost::phoenix::val;
namespace px = boost::phoenix;

namespace cnf_reader {

using literal_t = int;
using clause_t = std::vector<literal_t>;

struct cnf_t {
    size_t n_vars;
    size_t n_clauses;

    std::vector<clause_t> formula;

    void add_clause(std::vector<literal_t> const& clause)
    {
        formula.push_back(clause);
    }
};

/*
Example CNF format
p cnf 36 88
c Generated on Ursa with 8 bits and the following code:
9 -14 -29 0
-9 14 0
-9 29 0
10 -6 -7 0
-10 6 0
-10 7 0
-11 6 7 0
11 -6 0
11 -7 0
-12 6 7 0
-12 -6 -7 0
12 -6 7 0
12 6 -7 0
-13 10 16 0
13 -10 0
13 -16 0
*/

bool parse_cnf(std::istream& dimacs, cnf_t* cnf)
{
    using namespace boost::spirit::qi;
    namespace px = boost::phoenix;
    uint_parser<size_t> num_;

    auto eoil = eol | eoi;
    auto vars = px::ref(cnf->n_vars);
    auto clauses = px::ref(cnf->n_clauses);

    dimacs >> std::noskipws >> phrase_match(
        *(("c" >> *(char_ - eol) >> eol) | ("c" >> eol) | eol)
            >> ("p" >> lit("cnf") >> num_[vars = qi::_1] >> num_[clauses = qi::_1] >> eoil)
            >> *((*(int_ - lit(0)) >> lit(0) >> eol)[px::bind(&cnf_t::add_clause, cnf, qi::_1)])
            >> eoil,
        blank);

    return dimacs.good() || dimacs.eof();
}

} // namespace cnf_reader

void CnfManager::add_clause()
{
    formula_data->add_clause(current_clause_lits, current_clause_size);
}

void CnfManager::add_lit(Var v, bool sign)
{
#ifdef USE_ASSERTIONS
    assert(current_clause_size < MAX_CLAUSE_SIZE - 1);
#endif

    Lit l = mkLit(v, sign);

    current_clause_lits[current_clause_size] = l;
    current_clause_size++;

    if (v + 1 > max_var) {
        max_var = v + 1;
    }

    auto it = occurrences->find(v);

    if (it == occurrences->end()) {
        (*occurrences)[v] = 1;
    } else {
        (*occurrences)[v]++;
    }

    if ((*occurrences)[v] > max_occurrence_of_var) {
        max_occurrence_of_var = (*occurrences)[v];
        var_with_more_occurrences = v;
    }
}

bool CnfManager::read_cnf(const char* file, FormulaData& data)
{
    std::ifstream input(file);
    if (!input) {
        return false;
    }

    cnf_reader::cnf_t cnf;
    if (!cnf_reader::parse_cnf(input, &cnf)) {
        return false;
    }

    max_var = -1;
    var_with_more_occurrences = -1;
    max_occurrence_of_var = -1;

    formula_data = &data;

    set_header(cnf.n_vars, cnf.n_clauses);

    for (auto const& clause : cnf.formula) {
        start_new_clause();

        for (int parsed_lit : clause) {
            Var var = std::abs(parsed_lit) - 1;
            add_lit(var, parsed_lit > 0);
        }

        add_clause();
    }

    if (max_var > header_n_vars) {
        printf("header claims %d vars, "
               "but highest var found is %d. Using %d...",
            header_n_vars, max_var, max_var);
    }

    data.set_n_vars(max_var);

    if (formula_data->get_n_clauses() != header_n_clauses) {
        printf("header claims %d clauses, but found %d clauses. Using %d...",
            header_n_clauses, formula_data->get_n_clauses(),
            formula_data->get_n_clauses());
    }

    formula_data->copy_host_clauses_to_dev();
    formula_data->set_most_common_var(var_with_more_occurrences, max_occurrence_of_var);

    return true;
}
