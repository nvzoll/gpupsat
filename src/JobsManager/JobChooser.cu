#include "JobChooser.cuh"

#include <algorithm>

MaxClauseJobChooser::MaxClauseJobChooser(
    std::vector<Clause> const& formula,
    size_t n_vars,
    size_t n_dead_vars,
    size_t threads,
    size_t n_blocks)

    : var_chooser(formula, n_vars)
    , JobChooser()
    , n_working_vars { n_vars - n_dead_vars }
    , n_threads { threads }
    , n_blocks { n_blocks }
    , evaluated { false }
    , strategy { ChoosingStrategy::DISTRIBUTE_JOBS_PER_THREAD }
{

}

MaxClauseJobChooser::MaxClauseJobChooser(
    std::vector<Clause> const& formula,
    size_t n_vars,
    size_t n_dead_vars,
    size_t threads,
    size_t n_blocks,
    ChoosingStrategy strategy)

    : var_chooser(formula, n_vars)
    , JobChooser()
    , n_working_vars { n_vars - n_dead_vars }
    , n_threads { threads }
    , n_blocks { n_blocks }
    , evaluated { false }
    , strategy { strategy }
{

}

MaxClauseJobChooser::~MaxClauseJobChooser()
{

}

bool MaxClauseJobChooser::is_evaluated()
{
    return evaluated;
}

void MaxClauseJobChooser::evaluate()
{
    evalVarPerJobs();

    chosen_vars.resize(vars_per_job);

    var_chooser.evaluate();

    for (size_t i = 0; i < vars_per_job; i++) {
#ifdef USE_ASSERTIONS
        assert(var_chooser.has_next_var());
#endif

        chosen_vars[i] = var_chooser.next_var();
    }

    evaluated = true;
}

void MaxClauseJobChooser::getJobs(JobsQueue& queue)
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif

    m_fixed_lits.resize(vars_per_job);
    addJobs(m_fixed_lits.data(), 0, queue);
}

void MaxClauseJobChooser::addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue& queue)
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

size_t MaxClauseJobChooser::get_n_jobs()
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif

    return static_cast<size_t>(std::pow(2, static_cast<double>(vars_per_job)));
}

void MaxClauseJobChooser::evalVarPerJobs()
{
    switch(strategy) {
    case ChoosingStrategy::DISTRIBUTE_JOBS_PER_THREAD:
        evalVarPerJobsDistribute();
        break;

    case ChoosingStrategy::UNIFORM:
        evalVarPerJobsUniform();

        break;
    default:
        assert(false);
        break;
    }
}

void MaxClauseJobChooser::evalVarPerJobsDistribute()
{
    double expected = (n_threads * n_blocks) * JOBS_PER_THREAD;
    double max = std::pow(2, std::min<size_t>(std::max<size_t>((n_working_vars - MIN_FREE_VARS), 1), 62));

#ifdef USE_ASSERTIONS
    assert(max > 0);
#endif

    expected = std::min(expected, max);

    vars_per_job = static_cast<size_t>(std::log2(expected)) + 1 ;
    vars_per_job = std::min<size_t>(vars_per_job, MAX_VARS);
}

void MaxClauseJobChooser::evalVarPerJobsUniform()
{
    size_t expected = UNIFORM_NUMBER_OF_VARS;
    size_t max = std::max<size_t>(n_working_vars - MIN_FREE_VARS, 1);

    vars_per_job = std::min<size_t>(std::min<size_t>(expected, max), MAX_VARS);
}
