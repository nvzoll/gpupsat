#ifndef __RESULTS_CUH__
#define __RESULTS_CUH__

#include "SATSolver.cuh"
#include <vector>

class Results {
public:
    Results(int n_vars, bool on_gpu);
    __device__ void set_satisfiable_results(Lit* results, int size);
    __host__ __device__ void set_undef();
    __host__ void set_host_status(sat_status status);
    __host__ void print_results(std::vector<Lit> const& solved_literals, std::vector<Clause> const& formula);
    __host__ sat_status get_status();
    // New host function to initialize pointers for device-side object template
    __host__ void init_device_pointers(sat_status* dev_results_array, sat_status* dev_formula_status, int num_vars);

private:
    /**
     * Results stored as sat_status rather than literals to allow having undefined literals!
     */
    sat_status* results_dev;
    sat_status* formula_status_dev;
    sat_status formula_status_host;
    int n_vars;
    bool on_gpu;

    __host__ void print_sat_results(sat_status, std::vector<Lit> const& solved_literals, std::vector<Clause> const& formula);
};

// Implementation of the new host function
inline __host__ void Results::init_device_pointers(sat_status* dev_results_array, sat_status* dev_formula_status, int num_vars)
{
    // This function is intended to be called ONLY on a temporary host object
    // that will be copied to the device. It sets the internal device pointers.
    this->results_dev = dev_results_array;
    this->formula_status_dev = dev_formula_status;
    this->n_vars = num_vars;
    // 'on_gpu' should have been set correctly by the constructor (called with true)
    // 'formula_status_host' is irrelevant for the device copy
}

#endif /* __RESULTS_CUH__ */
