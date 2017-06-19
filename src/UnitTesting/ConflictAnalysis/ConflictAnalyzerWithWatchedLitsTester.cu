#include "ConflictAnalyzerWithWatchedLitsTester.cuh"

__device__ ConflictAnalyzerWithWatchedLitsTester::ConflictAnalyzerWithWatchedLitsTester(
    DataToDevice& data, int input_file_number)
{
    this->data = &data;
    this->input_file_number = input_file_number;
}

__device__ ConflictAnalyzerWithWatchedLits ConflictAnalyzerWithWatchedLitsTester::generate_input(
    VariablesStateHandler& handler)
{
    ConflictAnalyzerWithWatchedLits cwwl(data->get_n_vars(), data->get_clauses_db_ptr(),
                                         &handler, true, data->get_max_implication_per_var(), nullptr, data->get_statistics_ptr(),
                                         data->get_nodes_repository_ptr());

    handler.set_assumptions(&assumptions);

    return cwwl;
}

__device__ bool ConflictAnalyzerWithWatchedLitsTester::test_propagate_1_file1(
    ConflictAnalyzerWithWatchedLits *&input)
{
    VariablesStateHandler handler(data->get_n_vars(), data->get_dead_vars_ptr(), nullptr);

    handler.set_decision_level(1);
    ConflictAnalyzerWithWatchedLits cwwl = generate_input(handler);

    input = &cwwl;

    Lit l = mkLit(3, false);

    Decision d;
    d.branched = false;
    d.decision_level = 1;
    d.literal = l;

    handler.new_decision(d);

    cwwl.propagate(d);

    if (handler.n_decisions() != 1) {
        printf("\tAdded decisions, when shouldn't\n\t");
        handler.print_decisions();
        return false;
    }

    int matched_implications = 0;

    Lit l1 = mkLit(4, true);
    Lit l2 = mkLit(5, true);
    Lit l3 = mkLit(6, true);

    for (int i = 0; i < handler.n_implications(); i++) {
        Decision *implication = handler.get_implication(i);
        Lit literal = implication->literal;

        if (literal == l1) {
            matched_implications++;
        }
        if (literal == l2) {
            matched_implications++;
        }
        if (literal == l3) {
            matched_implications++;
        }

    }

    if (matched_implications != 3) {
        printf("\tThe literals that should be implicated, were not\n");
        handler.print_implications();
        return false;
    }

    if (handler.n_implications() != 3) {
        printf("\tIt seams that some literals was implicated more than once.\n");
        handler.print_implications();
        return false;
    }

    return true;
}

__device__ bool ConflictAnalyzerWithWatchedLitsTester::test_propagate_2_file1(
    ConflictAnalyzerWithWatchedLits *&input)
{
    VariablesStateHandler handler(data->get_n_vars(), data->get_dead_vars_ptr(), nullptr);

    ConflictAnalyzerWithWatchedLits cawwl = generate_input(handler);

    Lit l = mkLit(8, false);
    Decision d;
    d.branched = false;
    d.decision_level = 1;
    d.literal = l;

    handler.new_decision(d);
    handler.set_decision_level(1);

    sat_status status = cawwl.propagate(d);

    bool contains = false;
    Lit l2 = mkLit(8, true);

    for (int i = 0; i < handler.n_decisions(); i++) {
        if (l2 == handler.get_decision(i).literal) {
            contains = true;
        }
    }
    for (int i = 0; i < handler.n_implications(); i++) {
        if (l2 == handler.get_implication(i)->literal) {
            contains = true;
        }
    }

    if (!contains) {
        printf("Literal ");
        print_lit(l2);
        printf(" should have been learnt, but is was not.\n");
        handler.print_all();
        return false;
    }

    if (status != sat_status::UNDEF) {
        printf("Status should be sat_status::UNDEF, but instead it was ");
        print_status(status);
        printf("\n");
    }

    return true;

}

__device__ bool ConflictAnalyzerWithWatchedLitsTester::test_implicate_and_backtrack_file1()
{
    return true;
}

__device__ bool ConflictAnalyzerWithWatchedLitsTester::test_reset_file1()
{

    VariablesStateHandler handler(data->get_n_vars(), data->get_dead_vars_ptr(), nullptr);

    ConflictAnalyzerWithWatchedLits test1 = generate_input(handler);

    return true;
}

__device__ bool ConflictAnalyzerWithWatchedLitsTester::test_propagate_all_file2()
{
    VariablesStateHandler handler(data->get_n_vars(), data->get_dead_vars_ptr(), nullptr);
    ConflictAnalyzerWithWatchedLits test1 = generate_input(handler);

    Decision d1, d2, i1;

    d1.decision_level = 1;
    d1.literal = mkLit(15, true);
    d1.branched = false;

    d2.decision_level = 2;
    d2.literal = mkLit(38, true);
    d2.branched = false;

    i1.decision_level = 0;
    i1.literal = mkLit(41, false);
    i1.implicated_from_formula = true;

    handler.new_implication(i1);
    test1.propagate(i1);
    handler.print_all();

    handler.increment_decision_level();

    handler.new_decision(d1);
    test1.propagate(d1);

    handler.increment_decision_level();

    handler.new_decision(d2);
    test1.propagate(d2);

    //~V11(1) ~V12(1) ~V16(1) ~V17(1) ~V34(1) ~V40(1) ~V7(1)
    //~V35(1) ~V25(1) ~V29(1) ~V28(1) ~V37(1) ~V0(1) ~V6(1)
    //~V23(1) V24(2) ~V30(2) ~V36(2) ~V18(2) V31(2) ~V1(2) ~V13(2)
    //~V19(2) V22(2) ~V10(2) ~V4(2) V5(2)
    Var other_imp_vars[27] = {11, 12, 16, 17, 34, 40,
                              7, 35, 25, 29, 28, 37, 0, 6, 23, 24, 30, 36,
                              18, 31, 1, 13, 19, 22, 10, 4, 5
                             } ;
    bool other_imp_signs[27] = {false, false, false, false,
                                false, false, false, false, false, false, false,
                                false, false, false, false, true, false, false,
                                false, true, false, false, false, true, false, false, true
                               };

    for (int i = 0; i < 27; i++) {
        if (handler.is_var_free(i)) {
            Decision implication;
            implication.implicated_from_formula = false;
            implication.literal = mkLit(other_imp_vars[i], other_imp_signs[i]);
            if (i < 15) {
                implication.decision_level = 1;
                handler.set_decision_level(1);
            }
            else {
                implication.decision_level = 2;
                handler.set_decision_level(2);
            }

            handler.new_implication(implication);
            test1.propagate(implication);



        }
    }

    handler.print_all();


    return true;

}

__device__ void ConflictAnalyzerWithWatchedLitsTester::test_all()
{
    printf("ConflictAnalyzerWithWatchedLits tester:\n");
    switch(input_file_number) {
    case 1:
        ConflictAnalyzerWithWatchedLits *dummy;
        Tester::process_test(test_propagate_1_file1(dummy), "Test Propagate 1");
        Tester::process_test(test_propagate_2_file1(dummy), "Test Propagate 2");
        Tester::process_test(test_implicate_and_backtrack_file1(), "Test Propagate 2");
        Tester::process_test(test_reset_file1(), "Test Reset");
        break;
    case 2:
        Tester::process_test(test_propagate_all_file2(), "Test propagate all");
        break;
    }
}
