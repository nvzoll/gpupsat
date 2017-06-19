/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include "PropagationTester.cuh"

__device__ PropagationTester::PropagationTester(
    const CUDAClauseVec *f,
    VariablesStateHandler *vh)
    : formula { f }
    , vars_handler { vh }
{

}

__device__ bool PropagationTester::test_single_propagation(ConflictAnalyzer *analyzer,
        sat_status& status)
{
    if (vars_handler->no_free_vars()) {
        return true;
    }

    int initial_implics = vars_handler->n_implications();

    int new_dec_level = vars_handler->get_decision_level() + 1;
    vars_handler->set_decision_level(new_dec_level);

    Decision d;
    d.decision_level = new_dec_level;
    d.literal = mkLit(vars_handler->first_free_var(),
                      new_dec_level % 2 == 0);

    vars_handler->new_decision(d);
    status = analyzer->propagate(d);

    if (status == sat_status::UNDEF) {
        int current_implics = vars_handler->n_implications();

        if (current_implics < initial_implics) {
            printf("\tNo conflict, but less implications after propagation!\n");
            return false;
        }

        for (int i = initial_implics; i < current_implics; i++) {
            Decision implication = *vars_handler->get_implication(i);

            if (!test_implication(implication,
                                  analyzer)) {
                printf("Decision ");
                print_decision(d);
                printf(" had implication ");
                print_decision(implication);
                printf(", but it could not be implicated!\n");


                return false;
            }
        }
    }

    return true;

}

__device__ bool PropagationTester::test_implication(
    Decision implication, ConflictAnalyzer *analyzer)
{

    for (int i = 0; i < formula->size_of(); i++) {
        Clause c = formula->get(i);

        bool impl_present = false;
        int other_lits_unsat = 0;

        for (int j = 0; j < c.n_lits; j++) {
            Lit l = c.literals[j];

            if (~l == implication.literal) {
                break;
            }

            if (l == implication.literal) {
                impl_present = true;
            }
            else {
                sat_status status = vars_handler->literal_status(l);
                if (status == sat_status::UNDEF || status == sat_status::SAT) {
                    break;
                }
                if (status == sat_status::UNSAT) {
                    other_lits_unsat++;
                }
            }

        }

        if (impl_present && other_lits_unsat == (c.n_lits - 1)) {
            return true;
        }

    }

    return false;

}
