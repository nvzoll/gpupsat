#include "JobChooser.cuh"
#include "MaxClauseJobChooser.cuh" // Include the new header

#include <algorithm>
#include <cmath>   // Include for std::pow, std::log2
#include <cassert> // Include for assert

MaxClauseJobChooser::MaxClauseJobChooser(
    std::vector<Clause> const &formula,
    size_t n_vars,
    size_t n_dead_vars,
    size_t threads,
    size_t n_blocks)

    : var_chooser(formula, n_vars), JobChooser(), n_working_vars{n_vars - n_dead_vars}, n_threads{threads}, n_blocks{n_blocks}, evaluated{false}, vars_per_job{0} // Initialize
      ,
      strategy{ChoosingStrategy::DISTRIBUTE_JOBS_PER_THREAD}
{
}

MaxClauseJobChooser::MaxClauseJobChooser(
    std::vector<Clause> const &formula,
    size_t n_vars,
    size_t n_dead_vars,
    size_t threads,
    size_t n_blocks,
    ChoosingStrategy strategy)

    : var_chooser(formula, n_vars), JobChooser(), n_working_vars{n_vars - n_dead_vars}, n_threads{threads}, n_blocks{n_blocks}, evaluated{false}, vars_per_job{0} // Initialize
      ,
      strategy{strategy}
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
    evalVarPerJobs(); // Determines vars_per_job

    chosen_vars.resize(vars_per_job);

    var_chooser.evaluate(); // Evaluate the underlying variable chooser

    // Select the top 'vars_per_job' variables
    for (size_t i = 0; i < vars_per_job; i++)
    {
#ifdef USE_ASSERTIONS
        assert(var_chooser.has_next_var());
#endif
        if (!var_chooser.has_next_var())
        {
            // Handle case where not enough variables are available
            vars_per_job = i; // Adjust vars_per_job to actual count
            chosen_vars.resize(vars_per_job);
            printf("Warning: Not enough variables available for MaxClauseJobChooser. Adjusted vars_per_job to %zu\n", vars_per_job);
            break;
        }
        chosen_vars[i] = var_chooser.next_var();
    }

    evaluated = true;
}

void MaxClauseJobChooser::getJobs(JobsQueue &queue)
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif
    if (!evaluated)
        return;

    m_fixed_lits.resize(vars_per_job);
    addJobs(m_fixed_lits.data(), 0, queue);
}

// Modified addJobs for MaxClauseJobChooser
void MaxClauseJobChooser::addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue &queue)
{
    // Base case: A job is complete
    if (n_fixed_lits == vars_per_job)
    {
        // Use the new queue::add method
        queue.add(fixed_lits, n_fixed_lits);
        return;
    }

    // Recursive step
    Lit lit_true = mkLit(chosen_vars[n_fixed_lits], true);
    fixed_lits[n_fixed_lits] = lit_true;
    addJobs(fixed_lits, n_fixed_lits + 1, queue);

    Lit lit_false = mkLit(chosen_vars[n_fixed_lits], false);
    fixed_lits[n_fixed_lits] = lit_false;
    addJobs(fixed_lits, n_fixed_lits + 1, queue);
}

size_t MaxClauseJobChooser::get_n_jobs()
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif
    if (!evaluated)
        return 0;
    return static_cast<size_t>(std::pow(2.0, static_cast<double>(vars_per_job)));
}

// Implementation of estimateTotalLiteralSize for MaxClauseJobChooser (removed const)
size_t MaxClauseJobChooser::estimateTotalLiteralSize()
{
#ifdef USE_ASSERTIONS
    assert(evaluated); // Should be evaluated before calling this
#endif
    if (!evaluated)
        return 0;
    // Each job has 'vars_per_job' literals
    // Restore original calculation
    size_t num_jobs = static_cast<size_t>(std::pow(2.0, static_cast<double>(vars_per_job)));
    return vars_per_job * num_jobs;
}

void MaxClauseJobChooser::evalVarPerJobs()
{
    switch (strategy)
    {
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
    // Ensure vars_per_job is not negative or excessively large after calculation
    if (vars_per_job > n_working_vars)
    {
        vars_per_job = n_working_vars;
    }
    if (vars_per_job > MAX_VARS)
    { // Assuming MAX_VARS is defined
        vars_per_job = MAX_VARS;
    }
    if (vars_per_job < 0)
    { // Should not happen with size_t, but defensive check
        vars_per_job = 0;
    }
}

void MaxClauseJobChooser::evalVarPerJobsDistribute()
{
    double expected_jobs = static_cast<double>(n_threads * n_blocks) * JOBS_PER_THREAD;
    // Calculate max possible jobs based on available working variables (2^n_working_vars)
    // Be careful with large n_working_vars causing overflow with std::pow
    double max_possible_jobs = (n_working_vars < 63) ? std::pow(2.0, static_cast<double>(n_working_vars)) : std::numeric_limits<double>::max();

#ifdef USE_ASSERTIONS
    assert(max_possible_jobs > 0);
#endif

    // Target jobs cannot exceed the maximum possible jobs
    expected_jobs = std::min(expected_jobs, max_possible_jobs);

    // Calculate vars needed for expected_jobs (log base 2)
    // Add 1 because log2(N) gives exponent E where 2^E = N. We need E vars for N jobs.
    // Handle expected_jobs <= 0 case for log2.
    if (expected_jobs <= 1.0)
    {
        vars_per_job = (expected_jobs == 1.0 && n_working_vars > 0) ? 1 : 0;
    }
    else
    {
        vars_per_job = static_cast<size_t>(std::log2(expected_jobs));
        // If expected_jobs is not a perfect power of 2, log2 truncates, potentially needing one more var.
        if (std::pow(2.0, static_cast<double>(vars_per_job)) < expected_jobs)
        {
            vars_per_job++;
        }
    }

    // Ensure vars_per_job doesn't exceed available working vars or MAX_VARS limit
    vars_per_job = std::min<size_t>(vars_per_job, n_working_vars);
    vars_per_job = std::min<size_t>(vars_per_job, MAX_VARS); // Assuming MAX_VARS is defined
}

void MaxClauseJobChooser::evalVarPerJobsUniform()
{
    size_t expected = UNIFORM_NUMBER_OF_VARS;
    // Max vars we can actually use is limited by working vars
    size_t max_usable_vars = n_working_vars; // No need to subtract MIN_FREE_VARS here

    vars_per_job = std::min<size_t>(expected, max_usable_vars);
    vars_per_job = std::min<size_t>(vars_per_job, MAX_VARS); // Assuming MAX_VARS is defined
}
