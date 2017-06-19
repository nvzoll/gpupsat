#include "ConflictAnalyzerGenericTester.cuh"

__device__ ConflictAnalyzerGenericTester::
ConflictAnalyzerGenericTester(DataToDevice& data,
                              VariablesStateHandler *handler)
    : conflictAnalyzer(data.get_n_vars(),
                       data.get_clauses_db_ptr(),
                       handler, true,
                       data.get_max_implication_per_var(),
                       nullptr,
                       data.get_statistics_ptr())

    , conflictAnalyzer_fullsearch(data.get_n_vars(),
                                  data.get_clauses_db_ptr(),
                                  handler, data.get_max_implication_per_var(),
                                  nullptr,
                                  data.get_statistics_ptr())

    , conflictAnalyzer_two_literals(data.get_n_vars(),
                                    data.get_clauses_db_ptr(),
                                    handler, true, data.get_max_implication_per_var(),
                                    nullptr,
                                    data.get_statistics_ptr(),
                                    data.get_nodes_repository_ptr())

    , vars_handler { handler }
    , n_clauses { data.get_clauses_db().size_of() }
    , n_vars { data.get_n_vars() }
    , formula { data.get_clauses_db_ptr() }
{
    handler->set_assumptions(&assumptions);

    conflictAnalyzer.set_assumptions(&assumptions);
    conflictAnalyzer_fullsearch.set_assumptions(&assumptions);
    conflictAnalyzer_two_literals.set_assumptions(&assumptions);
}

__device__ bool ConflictAnalyzerGenericTester::change_analyzers_states()
{
    vars_handler->backtrack_to(-1);
    conflictAnalyzer.reset();
    conflictAnalyzer_fullsearch.reset();
    conflictAnalyzer_two_literals.reset();

    int decision_level = 1;

    for (int i = 0; i < n_vars; i++) {
        if (i % 2 == 0 && vars_handler->is_var_free(i)) {
            vars_handler->set_decision_level(decision_level);
            Decision d;
            d.decision_level = decision_level;
            d.literal = mkLit(i, i % 4 == 0);
            d.implicated_from_formula = false;
            vars_handler->new_decision(d);
            conflictAnalyzer.make_decision(d);
            conflictAnalyzer_fullsearch.make_decision(d);
            conflictAnalyzer_two_literals.make_decision(d);
            decision_level++;
        }
    }

    return true;
}

__device__ bool ConflictAnalyzerGenericTester::test_init_states()
{
    bool (ConflictAnalyzerGenericTester::*funct_ptr)(ConflictAnalyzer * analyzer)
        = &ConflictAnalyzerGenericTester::test_init_state;

    return perform_tests(funct_ptr);

}
__device__ bool ConflictAnalyzerGenericTester::test_init_state(
    ConflictAnalyzer *analyzer)
{
    return analyzer->check_consistency();
}

__device__ bool ConflictAnalyzerGenericTester::test_reset(
    ConflictAnalyzer *analyzer)
{
    analyzer->reset();

    if (analyzer->get_use_implication_graph()) {
        for (int i = 0; i < n_vars; i++) {
            if (analyzer->is_set_in_graph(i)) {
                printf("\tReset the analyzer, "
                       "but its graph still contains the var %d\n", i);
                return false;
            }
        }
    }


    return analyzer->check_consistency();
}

__device__ bool ConflictAnalyzerGenericTester::perform_tests(
    bool (ConflictAnalyzerGenericTester::*funct_ptr)(ConflictAnalyzer *analyzer))
{
    bool tests = true;
    bool error_found_before = false;


    tests = tests && (*this.*funct_ptr)(&conflictAnalyzer);
    if (!tests) {
        printf("\tTesting Conflict Analyzer\n");
        error_found_before = true;
    }


    tests = tests && (*this.*funct_ptr)(&conflictAnalyzer_fullsearch);
    if (!tests && !error_found_before) {
        printf("\tTesting Conflict Analyzer Full Search\n");
        error_found_before = true;
    }

    tests = tests && (*this.*funct_ptr)(&conflictAnalyzer_two_literals);
    if (!tests && !error_found_before) {
        printf("\tTesting Conflict Analyzer Two Literals\n");
        error_found_before = true;
    }

    if (!conflictAnalyzer_two_literals.check_two_watched_literals_consistency()) {
        printf("Two watched literals is inconsistent!\n");
        tests = false;
    }

    return tests;
}



__device__ bool ConflictAnalyzerGenericTester::test_resets()
{
    bool (ConflictAnalyzerGenericTester::*funct_ptr)(ConflictAnalyzer * analyzer)
        = &ConflictAnalyzerGenericTester::test_reset;

    bool results = perform_tests(funct_ptr);

    vars_handler->reset();

    return results;
}

__device__ bool ConflictAnalyzerGenericTester::test_propagates()
{

    bool (ConflictAnalyzerGenericTester::*funct_ptr)(ConflictAnalyzer * analyzer)
        = &ConflictAnalyzerGenericTester::test_propagate;

    bool test =  perform_tests(funct_ptr);

    return test;
}



__device__ bool ConflictAnalyzerGenericTester::test_propagate(
    ConflictAnalyzer *analyzer)
{

    PropagationTester propag_tester(formula, vars_handler);

    sat_status status;

    bool results = true;
    int limit = 300;
    int run = 0;

    status = sat_status::UNDEF;

    while (status == sat_status::UNDEF && !vars_handler->no_free_vars() && run < limit) {
        results = results & propag_tester.test_single_propagation(analyzer, status);
        run++;
    }

    vars_handler->reset();
    analyzer->reset();

    return results;
}

__device__ void ConflictAnalyzerGenericTester::test_all()
{
    printf("Conflict analyzer generic tester:\n");

    Tester::process_test(test_init_states(), "Test init states");
    Tester::process_test(change_analyzers_states(), "Change analyzer's state");
    Tester::process_test(test_resets(), "Test resets");
    //Tester::process_test(change_analyzers_states(), "Change analyzer's state");

    //TODO This test requires further checking
    //Tester::process_test(test_propagates(), "Test propagates");

    print_summary();

}
