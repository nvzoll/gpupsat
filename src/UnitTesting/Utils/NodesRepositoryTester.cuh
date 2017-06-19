#ifndef __NODESREPOSITORYTESTER_CUH__
#define __NODESREPOSITORYTESTER_CUH__

#define NUMBER_OF_TESTS_TO_STRESS 500

#include "UnitTesting/Tester.cuh"
#include "Utils/NodesRepository.cuh"
#include "Utils/GPULinkedList.cuh"
#include "BCPStrategy/WatchedClausesList.cuh"

class NodesRepositoryTester : public Tester
{
private:
    __device__ bool stress_test();
    NodesRepository<GPULinkedList<WatchedClause *>::Node> *repository;
    __device__ bool is_pointer_valid(GPULinkedList<WatchedClause *>::Node *pointer);

public:
    __device__ NodesRepositoryTester(DataToDevice data);
    __device__ void test_all();
};

#endif /* __NODESREPOSITORYTESTER_CUH__ */
