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
enum class ChoosingStrategy
{
    DISTRIBUTE_JOBS_PER_THREAD,
    UNIFORM
};

#include "VariableChooser.cuh"
#include "SATSolver/JobsQueue.cuh"

class JobChooser
{
public:
    virtual void evaluate() = 0;
    virtual void getJobs(JobsQueue &queue) = 0;
    virtual size_t get_n_jobs() = 0;
    virtual bool is_evaluated() = 0;
    // Added pure virtual function for estimating total literals (removed const for diagnostics)
    virtual size_t estimateTotalLiteralSize() = 0;

protected: // Changed from private to allow override
    virtual void addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue &queue) = 0;
};

// MaxClauseJobChooser declaration moved to MaxClauseJobChooser.cuh

#endif /* __JOBCHOOSER_CUH__ */
