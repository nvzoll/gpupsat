#include "CUDAListGraphTester.cuh"
#include "UnitTesting/Utils/StringUtils.cuh"

__device__ CUDAListGraphTester::CUDAListGraphTester(DataToDevice& data)
    : Tester()
    , graph(data.get_n_vars(),
            //data.get_max_implication_per_var())
            data.get_n_vars() + 1)
    , n_vars { data.get_n_vars() }
    , n_clauses { data.get_clauses_db().size_of() }
{
    CUDA_strcpy(tester_name, "CUDAListGraphTester");
}

__device__ bool CUDAListGraphTester::test_initial_state()
{
    if (!graph.are_all_unflagged()) {
        printf("\tGraph started with flagged vertices!\n");
        return false;
    }

    for (int i = 0; i < n_vars + 1; i++) {
        if (graph.is_set(i)) {
            printf("\tVariable %d is set on initial graph!\n", i);
            return false;
        }

        if (graph.contains(i)) {
            printf("\tVariable %d is in initial graph "
                   "(and worse, it is not set, so 2 errors)!\n", i);
            return false;
        }

        if (i < n_vars && var(graph.get(i).literal) != i) {
            printf("\tVertex of var %d does not have literal of its var.", i);
            return false;
        }

    }

    if (!graph.check_consistency()) {
        return false;
    }

    return true;
}

__device__ bool CUDAListGraphTester::test_backtrack_to()
{

    int low_decision_level = -1;

    graph.backtrack_to(low_decision_level);

    for (int i = 0; i < n_vars + 1; i++) {
        if (graph.is_set(i)) {
            printf("\tIs set after backtracking all\n");
            return false;
        }
    }

    low_decision_level = 4;
    int low_decision_level_2 = 3;
    int high_decision_level = 5;
    int medium_decision_level = 4;

    int low_added = 0;
    int high_added = 0;

    for (int i = 0; i < n_vars; i++) {
        Decision d;
        d.literal = mkLit(i, false);
        d.implicated_from_formula = false;

        if (i % 2 == 0) {
            if (i % 4 == 0) {
                d.decision_level = low_decision_level;
            }
            else {
                d.decision_level = low_decision_level_2;
            }
            low_added++;
        }
        else {
            d.decision_level = high_decision_level;
            high_added++;
        }

        graph.set(d);
    }

    graph.backtrack_to(medium_decision_level);

    for (int i = 0; i < n_vars; i++) {
        if (graph.is_set(i)) {
            if (graph.get(i).decision_level != low_decision_level &&
                graph.get(i).decision_level != low_decision_level_2
               ) {
                printf("\tA vertex with high level was not removed in backtrack "
                       "with decision level = %d\n",
                       graph.get(i).decision_level);
                return false;
            }
            low_added--;
        }
        else {
            high_added--;
        }
    }

    if (low_added != 0) {
        printf("\tSome low decision level was removed in backtrack\n");
        return false;
    }
    if (high_added != 0) {
        printf("\tSome high decision level was not removed in backtrack\n");
        return false;
    }

    if (!graph.check_consistency()) {
        printf("\tNot consistent in the end\n");
        return false;
    }

    return true;
}
__device__ bool CUDAListGraphTester::test_set_and_is_set()
{

    for (int i = 0; i < n_vars; i++) {
        if (!graph.is_set(i)) {
            Decision d;
            Lit lit_added = mkLit(i, false);
            int dec_level_added = 3;
            d.literal = lit_added;
            d.decision_level = dec_level_added;
            d.implicated_from_formula = false;

            graph.set(d);

            if (!graph.is_set(i)) {
                printf("\tNot set after inserting\n");
                return false;
            }

            if (graph.get(i).literal != lit_added) {
                printf("\tLiterals set is not the one returned\n");
                return false;
            }

            if (graph.get(i).decision_level != dec_level_added) {
                printf("\tReturning decision level that does not match with added.\n");
                return false;
            }

        }
    }

    if (!graph.check_consistency()) {
        printf("\tNot consistent in the end\n");
        return false;
    }

    return true;

}

__device__ bool CUDAListGraphTester::test_link_and_neighbors_methods()
{
    // Emptying graph.
    graph.backtrack_to(-1);

    for (int i = 0; i < n_vars; i++) {
        Decision d;
        d.decision_level = 3;
        d.literal = mkLit(i, i % 2 == 0);
        d.implicated_from_formula = false;
        graph.set(d);
    }


    Clause c;
    Lit l1, l2;
    l1 = mkLit(0, true);
    l2 = mkLit(1, false);
    create_clause_on_dev(2, c);
    addLitToDev(l1, c);
    addLitToDev(l2, c);

    Decision d;
    d.literal = mkLit(n_vars, false);
    d.implicated_from_formula = false;
    graph.set(d);

    for (int i = 0; i < n_vars; i++) {
        for (int j = i + 1; j < n_vars; j++) {
            graph.link(i, j, &c);
        }
        graph.link_with_conflict(i, &c, 3);
    }

    for (int i = 0; i < n_vars + 1; i++) {
        int neighbors = graph.get_n_neighbors(i, false);
        if (neighbors != i) {
            printf("\tNumber of neighbors of vertex %d "
                   "is not the same as added(%d != %d)\n", i, neighbors, i);
            return false;
        }

        CUDAListGraph::Iterator iter = graph.get_iterator(i, false);

        while (iter.has_next()) {
            int index = iter.get_next_index();
            int neibor = graph.get_neighbor_index(i, index, false);

            if (neibor >= i) {
                printf("\tAdded wrong neighbor\n");
                return false;
            }
        }
    }

    if (!graph.check_consistency()) {
        return false;
    }

    return true;
}

__device__ bool CUDAListGraphTester::test_flag_unflag()
{
    graph.unflag_all();

    if (!graph.are_all_unflagged()) {
        printf("\tUnflagged all but not all are unflagged.\n");
        return false;
    }

    for (int i = 0; i < n_vars + 1; i++) {
        if (i % 2 == 0) {
            graph.flag(i);
        }
    }

    for (int i = 0; i < n_vars + 1; i++) {
        if (i % 3 == 0) {
            graph.unflag(i);
        }
    }

    for (int i = 0; i < n_vars + 1; i++) {
        if ((i % 2 == 0) && (i % 3 != 0)) {
            if (!graph.is_flagged(i)) {
                printf("\tA vertex that should be flagged is not flagged!\n");
                return false;
            }
        }
        else {
            if (graph.is_flagged(i)) {
                printf("\tA vertex that should not be flagged is flagged!\n");
                return false;
            }
        }
    }

    if (!graph.check_consistency()) {
        return false;
    }

    return true;
}

__device__ bool CUDAListGraphTester::test_link_linked()
{
    graph.reset();

    for (int i = 0; i < n_vars / 2; i++) {

        Var v1 = i * 2;
        Var v2 = i * 2 + 1;

        Decision d1;
        Decision d2;
        d1.implicated_from_formula = false;
        d2.implicated_from_formula = false;
        d1.literal = mkLit(v1, true);
        d2.literal = mkLit(v2, false);

        graph.set(d1);
        graph.set(d2);

        Clause c;
        create_clause_on_dev(2, c);

        addLitToDev(d1.literal, c);
        addLitToDev(d2.literal, c);

        graph.link(v1, v2, &c);

        graph.link_with_conflict(v1, &c, 4);

    }

    for (int i = 0; i < n_vars; i++) {
        for (int j = i; j < n_vars; j++) {
            bool present = graph.linked(i, j);

            if (i % 2 == 0 && j == i + 1) {
                if (!present) {
                    printf("\tThe vars %d and %d were linked,"
                           " but 'linked' returned false for them.\n", i, j);
                    return false;
                }
            }
            else {
                if (present) {
                    printf("\tThe vars %d and %d were not linked,"
                           " but 'linked' returned true for them.\n", i, j);
                    return false;
                }
            }
        }
        bool present = graph.linked_with_conflict(i);
        if (i % 2 == 0 && (n_vars % 2 == 0 || i < n_vars - 1)) {
            if (!present) {
                printf("\tThe vertices %d and k (conflict) were linked,"
                       " but 'linked' returned false for them.\n", i);
                return false;
            }
        }
        else {
            if (present) {
                printf("\tThe vertices %d and k (conflict) were not linked,"
                       " but 'linked' returned true for them.\n", i);
                return false;
            }
        }
    }

    if (!graph.check_consistency()) {
        printf("\tGraph is not consistent.\n");
        return false;
    }
    return true;

}
__device__ bool CUDAListGraphTester::stress_test()
{
    graph.reset();

    for (int i = 0; i < n_vars; i++) {
        Decision d;
        d.decision_level = 12;
        d.implicated_from_formula = false;
        d.literal = mkLit(i, true);
        graph.set(d);
    }

    Clause c;
    create_clause_on_dev(1, c);
    Lit l = mkLit(0, true);
    addLitToDev(l, c);

    for (int i = 0; i < n_vars; i++) {
        for (int j = 0; j < n_vars; j++) {
            if (i != j) {
                graph.link(i, j, &c);
            }
        }
        graph.link_with_conflict(i, &c, 32);
    }

    for (int i = 0; i < n_vars; i++) {
        for (int j = 0; j < n_vars; j++) {
            if (i != j) {
                if (!graph.linked(i, j)) {
                    printf("Variables %d and %d were linked, but are not now.\n", i, j);
                    return false;
                }
            }
        }

        if (!graph.linked_with_conflict(i)) {
            printf("Variables %d and k (conflict were linked, but are not now.\n", i);
            return false;
        }

    }

    if (!graph.check_consistency()) {
        printf("Graph is not consistent.\n");
        return false;
    }

    return true;

}

__device__ void CUDAListGraphTester::test_all()
{

    printf("CUDAListGraph tester:\n");

    Tester::process_test(test_initial_state(), "initial state");
    Tester::process_test(test_set_and_is_set(), "set and is set");
    Tester::process_test(test_backtrack_to(), "backtrack to");
    Tester::process_test(test_link_and_neighbors_methods(),
                         "link and neighbors methods");
    Tester::process_test(test_flag_unflag(), "flag/unflag");
    Tester::process_test(test_link_linked(), "link/linked");
    //Tester::process_test(stress_test(), "Stress test");

    print_summary();

}
