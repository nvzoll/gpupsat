#include "LearntClausesManagerTester.cuh"

__device__ LearntClausesManagerTester::LearntClausesManagerTester(DataToDevice& data)
    : handler(data.get_n_vars(), data.get_dead_vars_ptr(), nullptr)
    , watched_clauses(data.get_n_vars(), &handler, data.get_nodes_repository_ptr(), data.get_clauses_db().size_of())
    , manager(&watched_clauses)
{
    //watched_clauses.add_all_clauses(*(data.get_clauses_db_ptr()));
    this->n_vars = data.get_n_vars();
    this->handler.set_assumptions(&assumptions);
}
__device__ bool LearntClausesManagerTester::add_random_clause(int size)
{
    Clause c = generate_clause(size);
    manager.learn_clause(c);
    return check_clause_in_structure(c);
}
__device__ Clause LearntClausesManagerTester::generate_clause(int size)
{
    Clause c;
    create_clause_on_dev(size, c);

    for (int i = 0; i < size; i++) {
        Var var;
        do {
            long long int value = clock64() + i * size;
            var =  (int)(value % n_vars);
        }
        while (c.contains(var));

        bool s = clock64() % 1000000 > 500000;
        Lit l = mkLit(var, s);
        addLitToDev(l, c);
    }

    return c;
}

__device__ bool LearntClausesManagerTester::check_clause_in_structure(Clause c)
{
    int in_var_lits = 0;

    for (int i = 0; i < c.n_lits; i++) {
        Var v = var(c.literals[i]);
        in_var_lits += watched_clauses.contains(v, c) ? 1 : 0;
    }

    if (watched_clauses.blocked_contains(c)) {
        if (in_var_lits != 0) {
            printf("\tClause ");
            print_clause(c);
            printf(" is in blocked clauses and %d other lists\n", in_var_lits);
            return false;
        }
    }
    else {
        if (in_var_lits != 2) {
            printf("\tClause ");
            print_clause(c);
            printf(", which is not blocked, should be in 2 lists of vars, "
                   "but it was in %d instead\n", in_var_lits);
            return false;
        }
    }
    return true;

}

__device__ bool LearntClausesManagerTester::stress_test()
{
    printf("Starting stress test, iterations: %d\n", manager.get_repository_capacity() * 2);
    //return true;
    for (int i = 0; i < manager.get_repository_capacity() * 2; i++) {
        int size = ((int)((clock64() + i) % 10)) + 2;
        printf("Size = %d\n", size);

        if (!add_random_clause(size)) {
            printf("\tError adding clause\n");
            return false;
        }

        manager.print_learnt_clauses_repository();

        //if (!watched_clauses.check_consistency())
        //return false;
    }

    return true;
}

__device__ void LearntClausesManagerTester::test_all()
{
    Tester::process_test(stress_test(), "Stress Test");
    //watched_clauses.print_structure();

}
