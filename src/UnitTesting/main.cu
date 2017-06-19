#include <vector>

#include "BCPStrategy/WatchedClausesListTester.cuh"
#include "ClauseLearning/LearntClausesManagerTester.cuh"
#include "ConflictAnalysis/ConflictAnalyzerGenericTester.cuh"
#include "ConflictAnalysis/ConflictAnalyzerWithWatchedLitsTester.cuh"
#include "ConflictAnalysis/CUDAListGraphTester.cuh"
#include "FileManager/CnfReader.h"
#include "FileManager/FileUtils.cuh"
#include "FileManager/FormulaData.cuh"
#include "SATSolver/BacktrackerTester.cuh"
#include "SATSolver/DataToDevice.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "SATSolver/VariablesStateHandler.cuh"
#include "Statistics/RuntimeStatisticsTester.cuh"
#include "UnitTesting/Tester.cuh"
#include "Utils/GPULinkedListTester.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "Utils/NodesRepositoryTester.cuh"

__global__ void unit_test_two_watched_conflict_analyzer(DataToDevice data, int file_number)
{
    ConflictAnalyzerWithWatchedLitsTester cawwlt(data, file_number);

    cawwlt.test_all();

    int errors = cawwlt.get_n_errors();
    int total = cawwlt.get_n_tests();

    printf("\nSummary:\nThere were %d errors out of %d total tests.\n", errors, total);

}

__global__ void unit_test_kernel(DataToDevice data)
{
    //data.get_clauses_db().print_all();

    CUDAListGraphTester graph_tester(data);
    VariablesStateHandler handler(data.get_n_vars(), data.get_dead_vars_ptr(), nullptr);
    BacktrackerTester backtracker_tester(data);
    WatchedClausesListTester watched_clauses_tester(data);
    LearntClausesManagerTester learnt_clauses_manager_tester(data);
    RuntimeStatisticsTester statistics_tester(data);
    NodesRepositoryTester nodes_repository_tester(data);
    GPULinkedListTester linked_list_tester;

    GPUStaticVec<Tester *, MAX_TESTERS> testers;

    /*
    testers.add(&graph_tester);
    testers.add(&backtracker_tester);
    testers.add(&watched_clauses_tester);
    testers.add(&statistics_tester);
    testers.add(&learnt_clauses_manager_tester);
    testers.add(&nodes_repository_tester);
    */

    testers.add(&linked_list_tester);

    int errors = 0;
    int total = 0;

    for (int i = 0; i < testers.size_of(); i++) {
        Tester *tester = testers.get(i);
        tester->test_all();

        errors += tester->get_n_errors();
        total += tester->get_n_tests();
    }

    printf("\nSummary:\nThere were %d errors out of %d total tests.\n", errors, total);

}

DataToDevice read_input(char *file)
{
    FormulaData fdata(n_lines(file), true);

    CnfManager().read_cnf(file, fdata);

    std::vector<Lit> set_lits = fdata.get_solved_literals();
    GPUVec<Var> dead_vars(set_lits.size());
    for (Lit lit : set_lits) {
        dead_vars.add(var(lit));
    }

    DataToDevice::numbers n = {
        fdata.get_n_vars(),
        fdata.get_n_clauses(),
        1, 1, 1,
        fdata.get_largest_clause_size()
    };

    DataToDevice::atomics atomics = { 0 };
    unsigned zero = 0;

    check(cudaMalloc(&atomics.next_job, sizeof(unsigned)), "Allocating counter");
    check(cudaMemcpy(atomics.next_job, &zero, sizeof(unsigned), cudaMemcpyHostToDevice),
        "Zeroing counter");

    check(cudaMalloc(&atomics.completed_jobs, sizeof(unsigned)), "Allocating counter");
    check(cudaMemcpy(atomics.completed_jobs, &zero, sizeof(unsigned), cudaMemcpyHostToDevice),
        "Zeroing counter");

    RuntimeStatistics stat(n.blocks, n.threads, atomics.completed_jobs);

    return DataToDevice(fdata.get_formula_dev(), dead_vars, stat, n, atomics);
}

void test_file(char *file)
{
    DataToDevice data_to_device = read_input(file);
    unit_test_kernel <<< 1, 1>>>(data_to_device);

}

int main(int argc, char **argv)
{

    if (argc < 2) {
        printf("Wrong number of parameters\n");
        exit(2);
    }

    if (argv[1][0] == '1') {
        printf("File = %s\n", argv[2]);
        test_file(argv[2]);
        cudaThreadSynchronize();
    }
    if (argv[1][0] == '2') {
        if (argc != 4) {
            printf("Wrong number of parameters\n");
            exit(2);
        }

        char *two_watched_test_file1 = argv[2];
        DataToDevice data_f1 = read_input(two_watched_test_file1);
        unit_test_two_watched_conflict_analyzer <<< 1, 1>>>(data_f1, atoi(argv[3]));
        cudaDeviceReset();
    }

    /*
    for (int i = 1; i < argc; i++)
    {
        test_file(argv[i]);
        cudaThreadSynchronize();
    }



    char* two_watched_test_file1 =
            "ConflictAnalyzerWithWatchedListTester_file1.cnf";

    DataToDevice data_f1 = read_input(two_watched_test_file1);



    char* two_watched_test_file2 =
                "ConflictAnalyzerWithWatchedListTester_file2.cnf";

    DataToDevice data_f2 = read_input(two_watched_test_file2);


    unit_test_two_watched_conflict_analyzer<<<1,1>>>(data_f2, 2);

    cudaDeviceReset();
    */
}
