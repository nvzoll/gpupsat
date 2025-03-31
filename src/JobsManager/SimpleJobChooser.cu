#include "SimpleJobChooser.cuh"

#include <algorithm>
#include <cmath>   // Include for std::pow
#include <cassert> // Include for assert

SimpleJobChooser::SimpleJobChooser(
    size_t n_vars,
    std::vector<Var> const &dv)

    : chosen_vars(), n_vars{n_vars}, dead_vars{dv}, evaluated{false}, vars_per_job{0} // Initialize vars_per_job
{
}

SimpleJobChooser::~SimpleJobChooser()
{
}

void SimpleJobChooser::evaluate()
{
    size_t total_vars = n_vars - dead_vars.size();
    // Use UNIFORM_NUMBER_OF_VARS from Configs.cuh (assuming it's included via JobChooser.cuh or similar)
    size_t to_choose = std::min<size_t>(total_vars, UNIFORM_NUMBER_OF_VARS);
    size_t n_chosen = 0;

    chosen_vars.clear(); // Clear previous choices if evaluate is called again
    for (size_t i = 0; n_chosen < to_choose && i < n_vars; i++)
    { // Added i < n_vars check
        Var v = static_cast<Var>(i);
        bool dead = false;

        // More efficient check if dead_vars is sorted
        // std::binary_search(dead_vars.begin(), dead_vars.end(), v);
        for (Var it : dead_vars)
        {
            if (it == v)
            {
                dead = true;
                break;
            }
        }

        if (!dead)
        {
            chosen_vars.push_back(v);
            n_chosen++;
        }

        // assert(i < n_vars); // Already checked in loop condition
    }

    vars_per_job = n_chosen; // Store the actual number chosen
    evaluated = true;
}
void SimpleJobChooser::getJobs(JobsQueue &queue)
{
#ifdef USE_ASSERTIONS
    assert(evaluated);
#endif
    if (!evaluated)
    {
        // Or handle error appropriately
        return;
    }

    // Resize the host-side temporary buffer
    m_fixed_lits.resize(vars_per_job);
    // Start the recursive job addition
    addJobs(m_fixed_lits.data(), 0, queue);
}

// Recursive function to generate jobs
void SimpleJobChooser::addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue &queue)
{
    // Base case: A complete assignment for the chosen variables is formed
    if (n_fixed_lits == vars_per_job)
    {
        // Directly add the literals from the host buffer to the queue's shared device buffer
        queue.add(fixed_lits, n_fixed_lits);
        return;
    }

    // Recursive step: Assign true and false to the next variable

    // Assign true
    Lit lit_true = mkLit(chosen_vars[n_fixed_lits], true);
    fixed_lits[n_fixed_lits] = lit_true;
    addJobs(fixed_lits, n_fixed_lits + 1, queue);

    // Assign false
    Lit lit_false = mkLit(chosen_vars[n_fixed_lits], false);
    fixed_lits[n_fixed_lits] = lit_false;
    addJobs(fixed_lits, n_fixed_lits + 1, queue);
}

size_t SimpleJobChooser::get_n_jobs()
{
#ifdef USE_ASSERTIONS
    assert(evaluated); // Should be evaluated before calling this
#endif
    if (!evaluated)
        return 0;
    // Calculate 2^vars_per_job
    return static_cast<size_t>(std::pow(2.0, static_cast<double>(vars_per_job)));
}

// New method implementation (removed const to match declaration)
size_t SimpleJobChooser::estimateTotalLiteralSize()
{
#ifdef USE_ASSERTIONS
    assert(evaluated); // Should be evaluated before calling this
#endif
    if (!evaluated)
        return 0;
    // Each job has 'vars_per_job' literals
    size_t num_jobs = static_cast<size_t>(std::pow(2.0, static_cast<double>(vars_per_job)));
    return vars_per_job * num_jobs;
}

bool SimpleJobChooser::is_evaluated()
{
    return evaluated;
}
