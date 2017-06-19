#include "BacktrackerTester.cuh"

__device__ BacktrackerTester::BacktrackerTester(DataToDevice& data) 
    : handler(data.get_n_vars(), data.get_dead_vars_ptr(), nullptr)
    , backtracker(&handler, nullptr, nullptr, data.get_statistics_ptr())
    , n_vars{ data.get_n_vars() }
{
    backtracker.set_watched_clauses(nullptr);

    handler.set_assumptions(&assumptions);
}

__device__ void BacktrackerTester::reset()
{
    handler.reset();
}
__device__ void BacktrackerTester::add_decisions(bool randomly_branched)
{
    int to_add = n_vars - 1;

    if (to_add < 0) {
        to_add = 1;
    }

    int dec_level = 1;

    for (int i = 0; i < to_add; i++) {
        if (handler.is_var_free(i)) {
            Decision d;
            if (randomly_branched) {
                d.branched = i % 3 == 0;
            }
            else {
                d.branched = false;
            }
            d.decision_level = dec_level;
            d.literal = mkLit(i, i % 2 == 0);
            handler.new_decision(d);
            dec_level++;
        }
    }
    handler.set_decision_level(dec_level - 1);
}

__device__ bool BacktrackerTester::test_non_chronological_backtracking()
{
    reset();
    add_decisions(false);

    int limit = 1000;
    int iters = 0;

    int last_dec_level = handler.get_decision_level();
    Decision last_decision = handler.get_last_decision();


    while (!(handler.no_decisions())) {
        sat_status status = backtracker.handle_chronological_backtrack();

        if (handler.get_decision_level() == 0) {
            if (status == sat_status::UNSAT) {
                return true;
            }
            else {
                printf("Backtracked to level 0 but did not return sat_status::UNSAT");
                return false;
            }
        }

        if (status != sat_status::UNDEF) {
            printf("Backtracking returned ");
            print_status(status);
            printf(" even though it did not backtrack to level 0\n");
            return false;
        }

        if (last_decision.branched) {
            if (last_dec_level - 1 != handler.get_decision_level()) {
                printf("Should have backtracked 1 levels, but didn't\n");
                return false;
            }
        }
        else {
            if (last_dec_level != handler.get_decision_level()) {
                printf("Should have only changed last decisions sign\n");
                return false;
            }
        }

        last_decision = handler.get_last_decision();
        last_dec_level = handler.get_decision_level();

        iters++;
        if (iters > limit) {
            printf("Limit of iterations reached but could not finish.\n");
            return false;
        }
    }

    return handler.get_decision_level() == 0;

}
__device__ bool BacktrackerTester::test_chronological_backtracking()
{
    reset();
    add_decisions(false);

    int limit = 1000;
    int iters = 0;

    int last_dec_level = handler.get_decision_level();
    Decision last_decision = handler.get_last_decision();

    while (!(handler.no_decisions())) {
        sat_status status =
            backtracker.handle_backtrack(handler.get_decision_level() - 3);

        if (handler.get_decision_level() == 0) {
            if (status == sat_status::UNSAT) {
                return true;
            }
            else {
                printf("Backtracked to level 0 but did not return sat_status::UNSAT");
                return false;
            }
        }

        if (last_dec_level - 3 != handler.get_decision_level()) {
            printf("Should have backtracked 3 levels, but didn't\n");
            return false;
        }

        last_decision = handler.get_last_decision();
        last_dec_level = handler.get_decision_level();

        iters++;
        if (iters > limit) {
            printf("Limit of iterations reached but could not finish.\n");
            return false;
        }
    }


    return handler.get_decision_level() <= 0;
}

__device__ void BacktrackerTester::test_all()
{
    Tester::process_test(test_non_chronological_backtracking(),
                         "Non-chronological backtracking");
    Tester::process_test(test_chronological_backtracking(),
                         "Chronological backtracking");
}
