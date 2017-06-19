#include "VSIDS.cuh"

__device__ VSIDS::VSIDS(size_t n_vars)
    : decay_factor { 2 }
    , clauses_before_decaying { 50 }
    , random_decision_frequency { 0.01f }
    , n_vars { n_vars }
    , vars { new vsids_var[n_vars] }
{
    for (size_t i = 0; i < n_vars; i++) {
        vars[i].free = true;
        vars[i].negative_lit_sum = 0;
        vars[i].positive_lit_sum = 0;
    }

    uint32_t seed = (uint32_t)clock();

    initCurand(seed);
}

__device__ void VSIDS::initCurand(uint32_t seed)
{
    curand_init(seed, 0, 0, &randState);
}

__device__ Var VSIDS::next_random_var()
{
    Var v;

    while (true) {
        v = curand(&randState) % n_vars;

        if (vars[v].free) {
            break;
        }
    }

    return v;
}

__device__ bool VSIDS::next_random_polarity()
{
    return curand(&randState) % 2 == 0;
}

__device__ void VSIDS::decay()
{
    for (int i = 0; i < n_vars; i++) {
        vars[i].negative_lit_sum /= decay_factor;
        vars[i].positive_lit_sum /= decay_factor;
    }
}

__device__ void VSIDS::increment(Lit literal)
{
    Var v = var(literal);
    bool s = sign(literal);

    if (s) {
        vars[v].positive_lit_sum++;
    }
    else {
        vars[v].negative_lit_sum++;
    }
}

__device__ void VSIDS::free_var(Var v)
{
#ifdef USE_ASSERTIONS
    assert(v >= 0 && v < n_vars);
#endif
    vars[v].free = true;
}

__device__ void VSIDS::block_var(Var v)
{
#ifdef USE_ASSERTIONS
    assert(v >= 0 && v < n_vars);
#endif
    vars[v].free = false;
}

__device__ void VSIDS::handle_clause(Clause c)
{
    for (int i = 0; i < c.n_lits; i++) {
        Lit literal = c.literals[i];
        increment(literal);
    }

    n_learnt_clauses++;

    if (n_learnt_clauses % clauses_before_decaying == 0
        && n_learnt_clauses > 0) {
        decay();
    }
}

__device__ Lit VSIDS::next_random_literal()
{
    return mkLit(next_random_var(), next_random_polarity());
}

__device__ Lit VSIDS::next_higher_literal()
{
    Lit higher;
    higher.x = -1;

    int last_sum = -1;

    for (size_t i = 0; i < n_vars; i++) {
        vsids_var vvar = vars[i];

        if (vvar.free) {
            if (vvar.positive_lit_sum > last_sum) {
                Lit l = mkLit(i, true);
                last_sum = vvar.positive_lit_sum;
                higher = l;
            }
            if (vvar.negative_lit_sum > last_sum) {
                Lit l = mkLit(i, false);
                last_sum = vvar.negative_lit_sum;
                higher = l;
            }

        }

    }
#ifdef USE_ASSERTIONS
    assert(higher.x >= 0);
#endif

    return higher;
}

__device__ Lit VSIDS::next_literal()
{
    float prop = curand_uniform(&randState);

    if (prop < random_decision_frequency) {
        return next_random_literal();
    }

    //if (n_decisions % clauses_before_decaying == 0 && n_decisions > 0)
    //    decay();

    n_decisions++;

    return next_higher_literal();
}

__device__ bool VSIDS::is_free(Var v)
{
    return vars[v].free;
}

__device__ void VSIDS::print()
{
    printf("Evals[%d] = {", n_vars);
    for (int i = 0; i < n_vars; i++) {
        Lit lp = mkLit(i, true);
        Lit ln = mkLit(i, false);

        print_lit(lp);
        printf("=%d ", vars[i].positive_lit_sum);
        print_lit(ln);
        printf("=%d(%s), ", vars[i].negative_lit_sum, vars[i].free ? "T" : "F");

    }

    printf("}\nDecisions = %d\n", n_decisions);
}
