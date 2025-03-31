#ifndef __MAXCLAUSEJOBCHOOSER_CUH__
#define __MAXCLAUSEJOBCHOOSER_CUH__

#include <vector>
#include <cstddef> // For size_t

#include "JobChooser.cuh" // Include the base class definition
#include "VariableChooser.cuh"
// #include "SATSolver/JobsQueue.cuh" // Included via JobChooser.cuh
// #include "SATSolver/SolverTypes.cuh" // Included via JobsQueue.cuh/VariableChooser.cuh? Check dependencies if needed.

class MaxClauseJobChooser : public JobChooser
{
public:
    MaxClauseJobChooser(
        std::vector<Clause> const &formula,
        size_t n_vars,
        size_t n_dead_vars,
        size_t n_threads,
        size_t n_blocks);

    MaxClauseJobChooser(
        std::vector<Clause> const &formula,
        size_t n_vars,
        size_t n_dead_vars,
        size_t threads,
        size_t n_blocks,
        ChoosingStrategy strategy);

    ~MaxClauseJobChooser();

    // Overrides from JobChooser
    void evaluate() override;
    void getJobs(JobsQueue &queue) override;
    size_t get_n_jobs() override;
    bool is_evaluated() override;
    size_t estimateTotalLiteralSize() override; // Removed const for diagnostics

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

    // Override from JobChooser (private in base, should match)
    void addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue &queue) override;

    // Helper methods
    void evalVarPerJobs();
    void evalVarPerJobsDistribute();
    void evalVarPerJobsUniform();
};

#endif /* __MAXCLAUSEJOBCHOOSER_CUH__ */