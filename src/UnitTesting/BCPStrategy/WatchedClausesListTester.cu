#include "WatchedClausesListTester.cuh"

__device__ WatchedClausesListTester::WatchedClausesListTester(DataToDevice& data)
    : handler(data.get_n_vars(), data.get_dead_vars_ptr(), nullptr)
    , watched_clauses_list(
        data.get_n_vars(), 
        &handler, 
        data.get_nodes_repository_ptr(),
        data.get_clauses_db().size_of())
{

    handler.set_assumptions(&assumptions);

    formula = data.get_clauses_db_ptr();
    watched_clauses_list.add_all_clauses(*formula);
}

__device__ void WatchedClausesListTester::test_all()
{
    Tester::process_test(test_implication_from_clauses(), "Test Implication from Clauses");
}
__device__ bool WatchedClausesListTester::test_implication_from_clauses()
{
    for (int i = 0; i < formula->size_of(); i++) {
        Clause c = formula->get(i);

        if (c.n_lits > 1) {

            watched_clauses_list.reset();
            handler.reset();

            const Clause *conflicting;
            GPULinkedList<found_implication> implications;

            sat_status status;

            for (int j = 0; j < c.n_lits - 1; j++) {
                GPULinkedList<found_implication>::LinkedListIterator
                iter = implications.get_iterator();
                bool is_unsat = false;
                while (iter.has_next()) {
                    found_implication impl = iter.get_next();

                    if (impl.implication.literal == ~(c.literals[c.n_lits - 1])) {
                        status = sat_status::UNSAT;
                        is_unsat = true;
                        break;
                    }

                }

                if (is_unsat) {
                    break;
                }



                Decision d;
                d.decision_level = j + 1;
                d.implicated_from_formula = false;
                d.literal = ~(c.literals[j]);

                handler.new_decision(d);

                status = watched_clauses_list.new_decision(d, &conflicting, implications);

                //if (j != c.n_lits - 2)
                //    implications.clear();

                if (status != sat_status::UNDEF) {
                    break;
                }


            }

            GPULinkedList<found_implication>::LinkedListIterator iter = implications.get_iterator();

            bool contains = false;

            while (iter.has_next()) {
                found_implication impl = iter.get_next();

                if (impl.implication.literal == c.literals[c.n_lits - 1]) {
                    contains = true;
                }

            }

            if (!contains && status == sat_status::UNDEF) {
                handler.print_all();
                printf("\tClause ");
                print_clause(c);
                printf(" should have implicated ");
                print_lit(c.literals[c.n_lits - 1]);
                printf(", but didn't. Its status is: ");
                print_status(status);
                printf("\n");
                //watched_clauses_list.print_structure();
                return false;
            }

            implications.clear();

        }

    }
    return true;
}
