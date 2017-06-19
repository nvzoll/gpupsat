#include "SimpleJobChooser.cuh"
#include "FileManager/FormulaData.cuh"
#include "FileManager/CnfReader.h"

int main_test_jobs()
{
    char *file = "subsetsum_random_4_5_without_unary.cnf";

    FormulaData data(200, true);

    CnfManager().read_cnf(file, data);

    std::vector<Var> dead_vars;

    dead_vars.push_back(4);
    dead_vars.push_back(6);

    //SimpleJobChooser(vector <Clause> * formula, int n_vars,
    //        std::vector<Var> * dead_vars);
    const std::vector<Clause> *formula = data.get_formula_host();

    SimpleJobChooser sjc(data.get_n_vars(),
                         &(dead_vars));
    JobChooser *jc = &sjc;

    jc->evaluate();

    int jobs = jc->get_n_jobs();

    JobsQueue jq(jobs);

    sjc.getJobs(jq);

    jq.print_jobs();

}
