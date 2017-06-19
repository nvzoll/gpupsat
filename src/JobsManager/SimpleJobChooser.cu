#include "SimpleJobChooser.cuh"

#include <algorithm>

SimpleJobChooser::SimpleJobChooser(
    size_t n_vars,
    std::vector<Var> const& dv)

    : chosen_vars()
    , n_vars { n_vars }
    , dead_vars { dv }
    , evaluated { false }
{

}

SimpleJobChooser::~SimpleJobChooser()
{

}

void SimpleJobChooser::evaluate()
{
    size_t total_vars = n_vars - dead_vars.size();
    size_t to_choose = std::min<size_t>(total_vars, UNIFORM_NUMBER_OF_VARS);
    size_t n_chosen = 0;

    for (size_t i = 0; n_chosen < to_choose; i++) {
        Var v = static_cast<Var>(i);
        bool dead = false;

        for (Var it : dead_vars) {
            if (it == v) {
                dead = true;
                break;
            }
        }

        if (!dead) {
            chosen_vars.push_back(v);
            n_chosen++;
        }

        assert(i < n_vars);
    }

    vars_per_job = n_chosen;
    evaluated = true;

}
void SimpleJobChooser::getJobs(JobsQueue& queue)
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif

    m_fixed_lits.resize(vars_per_job);
    addJobs(m_fixed_lits.data(), 0, queue);
}

void SimpleJobChooser::addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue& queue)
{
    if (n_fixed_lits == vars_per_job) {
        Job job = mkJob_dev(fixed_lits, n_fixed_lits);
        queue.add(job);

        return;
    }

    Lit lit = mkLit(chosen_vars[n_fixed_lits], true);
    fixed_lits[n_fixed_lits] = lit;
    addJobs(fixed_lits, n_fixed_lits + 1, queue);

    lit = mkLit(chosen_vars[n_fixed_lits], false);
    fixed_lits[n_fixed_lits] = lit;
    addJobs(fixed_lits, n_fixed_lits + 1, queue);

}

size_t SimpleJobChooser::get_n_jobs()
{
    return static_cast<size_t>(std::pow(2, static_cast<double>(vars_per_job)));
}

bool SimpleJobChooser::is_evaluated()
{
    return evaluated;
}
