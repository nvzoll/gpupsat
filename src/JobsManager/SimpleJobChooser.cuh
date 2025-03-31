#ifndef __SIMPLEJOBCHOOSER_CUH__
#define __SIMPLEJOBCHOOSER_CUH__

#include <vector>
#include "JobChooser.cuh"
#include "SATSolver/SolverTypes.cuh"

class SimpleJobChooser : public JobChooser
{
public:
    SimpleJobChooser(
        size_t n_vars,
        std::vector<Var> const &dead_vars);

    ~SimpleJobChooser();

    void evaluate();
    void getJobs(JobsQueue &queue);
    size_t get_n_jobs();
    bool is_evaluated();
    size_t estimateTotalLiteralSize(); // Removed const to match base class (diagnostic)
private:
    void addJobs(Lit *fixed_lits, size_t n_fixed_lits, JobsQueue &queue);

private:
    std::vector<Var> const &dead_vars;
    size_t n_vars;
    std::vector<Var> chosen_vars;
    std::vector<Lit> m_fixed_lits;
    size_t vars_per_job;
    bool evaluated;
};

#endif /* __SIMPLEJOBCHOOSER_CUH__ */
