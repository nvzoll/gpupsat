#include "NodesRepositoryTester.cuh"

__device__ NodesRepositoryTester::NodesRepositoryTester(DataToDevice data)
{
    this->repository = data.get_nodes_repository_ptr();
}

__device__ bool NodesRepositoryTester::is_pointer_valid(
    GPULinkedList<WatchedClause *>::Node *pointer)
{
    return pointer != nullptr && repository->within_boundaries(pointer);
}

__device__ bool NodesRepositoryTester::stress_test()
{
    printf("starting stress test\n");
    for (int i = 0; i < NUMBER_OF_TESTS_TO_STRESS; i++) {
        GPULinkedList<WatchedClause *>::Node *node = repository->alloc_element();

        if (!is_pointer_valid(node)) {
            return false;
        }
    }
    return true;
}

__device__ void NodesRepositoryTester::test_all()
{
    printf("Nodes repository tester:\n");
    Tester::process_test(stress_test(), "Stress test");
    Tester::print_summary();
}
