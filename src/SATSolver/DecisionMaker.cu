#include "DecisionMaker.cuh"

__device__ DecisionMaker::DecisionMaker(
    const CUDAClauseVec *formula,
    size_t n_vars)
#ifdef USE_VSIDS
    : vsids(n_vars)
#endif
{
#ifdef USE_VSIDS
    for (int i = 0; i < formula->size_of(); i++) {
        vsids.handle_clause(formula->get(i));
    }
#endif
}

__device__ void DecisionMaker::free_var(Var v)
{
#ifdef USE_VSIDS
    vsids.free_var(v);
#endif
}
__device__ void DecisionMaker::block_var(Var v)
{
#ifdef USE_VSIDS
    vsids.block_var(v);
#endif
}

__device__ void DecisionMaker::handle_learnt_clause(Clause c)
{
#ifdef USE_VSIDS
    vsids.handle_clause(c);
#endif

    /*
    printf("Learning clause: ");
    print_clause(c);
    printf("\n");
    vsids.print();
    */

}

__device__ Lit DecisionMaker::new_literal()
{
#ifdef USE_VSIDS
    return vsids.next_literal();
#else
    Var newVar = vars_handler->last_free_var();
    Lit literal = mkLit(newVar, true);

    return literal;
#endif
}

__device__ Decision DecisionMaker::new_decision()
{
    Lit literal = new_literal();
    Decision d;
    d.literal = literal;
    d.decision_level = vars_handler->get_decision_level() + 1;
    d.branched = false;
    return d;
}


__device__ Decision DecisionMaker::decide()
{

#ifdef USE_ASSERTIONS
    assert(vars_handler->get_decision_level() >= 0);
    assert(!vars_handler->no_free_vars());
#endif

    Decision d =  new_decision();
#if defined(DEBUG) || defined(IMPLICATION_GRAPH_DEBUG)
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Next decision: ");
        print_lit(d.literal);
        printf("(%d)\n", d.decision_level);
    }
#endif

    vars_handler->new_decision(d);
    vars_handler->set_decision_level(vars_handler->get_decision_level() + 1);

    return d;


}
__device__ void DecisionMaker::backtrack()
{


}
