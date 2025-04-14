#ifndef __JOBCHOOSER_CUH__
#define __JOBCHOOSER_CUH__

#include "SATSolver/Configs.cuh"

/**
 * The enum ChoosingStrategy is used to define the strategy
 * the chooser will use. They are:
 *
 *   * DISTRIBUTE_JOBS_PER_THREAD: Generate as many jobs as defined in JOBS_PER_THREAD
 * for each thread (if possible, or as many as possible)
 *      * UNIFORM: Generate jobs using as many vars as defined in UNIFORM_NUMBER_OF_VARS (if possible,
 * or as many as possible)
 *
 */
enum class ChoosingStrategy { DISTRIBUTE_JOBS_PER_THREAD, UNIFORM };

#include "VariableChooser.cuh"
#include "SATSolver/JobsQueue.cuh"

class JobChooser
{
public:
    virtual void evaluate() = 0;
    virtual void getJobs(JobsQueue& queue) = 0;
    virtual size_t get_n_jobs() = 0;
    virtual bool is_evaluated() = 0;

private:
    virtual void addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue& queue) = 0;
};

class MaxClauseJobChooser : public JobChooser
{
public:
    MaxClauseJobChooser(
        std::vector<Clause> const& formula,
        size_t n_vars,
        size_t n_dead_vars,
        size_t n_threads,
        size_t n_blocks);

    MaxClauseJobChooser(
        std::vector<Clause> const& formula,
        size_t n_vars,
        size_t n_dead_vars,
        size_t threads,
        size_t n_blocks,
        ChoosingStrategy strategy);

    ~MaxClauseJobChooser();

    void evaluate();
    void getJobs(JobsQueue& queue);
    size_t get_n_jobs();
    bool is_evaluated();

private:
    VariableChooser var_chooser;
    size_t n_working_vars;
    bool evaluated;
    size_t n_threads;
    size_t n_blocks;
    std::vector<Var> chosen_vars;
    std::vector<Lit> m_fixed_lits;
    size_t vars_per_job;
    ChoosingStrategy strategy;

    void addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue& queue);
    void evalVarPerJobs();
    void evalVarPerJobsDistribute();
    void evalVarPerJobsUniform();
};

#endif /* __JOBCHOOSER_CUH__ */
