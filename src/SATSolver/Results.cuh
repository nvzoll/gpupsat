#ifndef __RESULTS_CUH__
#define __RESULTS_CUH__

#include "SATSolver.cuh"
#include <vector>

class Results
{
public:
    Results(int n_vars, bool on_gpu);
    __device__ void set_satisfiable_results(Lit *results, int size);
    __host__ __device__ void set_undef();
    __host__ void set_host_status(sat_status status);
    __host__ void print_results(std::vector<Lit> const& solved_literals, std::vector<Clause> const& formula);
    __host__ sat_status get_status();
private:
    /**
     * Results stored as sat_status rather than literals to allow having undefined literals!
     */
    sat_status *results_dev;
    sat_status *formula_status_dev;
    sat_status formula_status_host;
    int n_vars;
    bool on_gpu;

    __host__ void print_sat_results(sat_status, std::vector<Lit> const& solved_literals, std::vector<Clause> const& formula);
};

#endif /* __RESULTS_CUH__ */
