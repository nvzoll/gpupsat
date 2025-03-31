#include "Results.cuh"

#include <vector>

Results::Results(int n_vars, bool on_gpu)
    : on_gpu { on_gpu }
    , n_vars { n_vars }
{

    if (on_gpu) {
        check(cudaMalloc(&results_dev, sizeof(sat_status) * n_vars), "Allocating results on GPU");
        check(cudaMalloc(&formula_status_dev, sizeof(sat_status)), "Allocating results on GPU");

        sat_status undef_stat = sat_status::UNDEF;
        sat_status unsat_stat = sat_status::UNSAT;

        check(cudaMemcpy(formula_status_dev, &unsat_stat, sizeof(sat_status), cudaMemcpyHostToDevice),
            "Copying results to GPU");

        for (int i = 0; i < n_vars; i++) {
            check(cudaMemcpy(results_dev + i, &undef_stat, sizeof(sat_status), cudaMemcpyHostToDevice), "Copying results to GPU");
        }
    } else {
        formula_status_host = sat_status::UNSAT;
    }
}

__device__ void Results::set_satisfiable_results(Lit* the_results, int size)
{
    assert(on_gpu);

    *formula_status_dev = sat_status::SAT;

    for (int i = 0; i < size; i++) {
        Lit l = the_results[i];
        int index = var(l);

#ifdef USE_ASSERTIONS
        assert(index >= 0);
#endif

        sat_status res_sign = sign(l) ? sat_status::SAT : sat_status::UNSAT;

        results_dev[index] = res_sign;
    }
}

__host__ __device__ void Results::set_undef()
{
#ifndef __CUDA_ARCH__
    assert(!on_gpu);
    formula_status_host = sat_status::UNDEF;
#else
    assert(on_gpu);
    *formula_status_dev = sat_status::UNDEF;
#endif
}
__host__ void Results::set_host_status(sat_status status)
{
    assert(!on_gpu);
    formula_status_host = status;
}

__host__ void Results::print_results(
    std::vector<Lit> const& solved_literals,
    std::vector<Clause> const& formula)
{
    sat_status status;

    if (on_gpu) {
        check(cudaMemcpy(&status, formula_status_dev, sizeof(sat_status), cudaMemcpyDeviceToHost), "Copying results from GPU");
    } else {
        status = formula_status_host;
    }

    switch (status) {
    case sat_status::UNDEF:
        printf("UNDEFINED\n");
        break;
    case sat_status::UNSAT:
        printf("UNSATISFIABLE\n");
        break;
    case sat_status::SAT:
        printf("SATISFIABLE\n");
        print_sat_results(status, solved_literals, formula);
        break;
    default:
        assert(false);
        break;
    }
}

__host__ sat_status Results::get_status()
{
    sat_status status;

    if (on_gpu) {
        check(cudaMemcpy(&status, formula_status_dev, sizeof(sat_status), cudaMemcpyDeviceToHost), "Copying results from GPU");
    } else {
        status = formula_status_host;
    }

    return status;
}

__host__ void Results::print_sat_results(
    sat_status status,
    std::vector<Lit> const& solved_literals,
    std::vector<Clause> const& formula)
{
    std::vector<sat_status> results_host(n_vars);

    if (on_gpu) {
        check(cudaMemcpy(results_host.data(), results_dev,
                  results_host.size() * sizeof(sat_status), cudaMemcpyDeviceToHost),
            "Copying results from GPU");
    } else {
        for (int i = 0; i < n_vars; i++) {
            results_host[i] = sat_status::UNDEF;
        }
    }

    for (Lit lit : solved_literals) {
        int index = var(lit);
        sat_status stat = sign(lit) ? sat_status::SAT : sat_status::UNSAT;
        results_host[index] = stat;
    }

    for (int i = 0; i < n_vars; i++) {
        if (results_host[i] == sat_status::SAT || results_host[i] == sat_status::UNDEF) {
            printf("%d ", i + 1);
        } else {
            if (results_host[i] == sat_status::UNSAT) {
                printf("-%d ", i + 1);
            }
        }
    }

    printf("0\n");

    bool eval = false;
    for (Clause const& cl : formula) {
        bool cl_eval = true;
        for (size_t i = 0; i < cl.n_lits; ++i) {
            int index = var(cl.literals[i]);
            cl_eval &= (results_host[index] == sat_status::SAT
                || results_host[index] == sat_status::UNDEF);
        }
        eval |= cl_eval;
    }

    printf("Solution ");
    printf((eval) ? "was verified\n" : "was not verified\n");
}
