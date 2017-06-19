#include "GPULinkedListTester.cuh"

__device__ GPULinkedListTester::GPULinkedListTester()
{

}

__device__ bool GPULinkedListTester::stress_test()
{
    GPULinkedList<int> list;

    for (int i = 0; i < NUMBER_OF_TESTS; i++) {
        list.push_back(i);
    }

    GPULinkedList<int>::LinkedListIterator iter = list.get_iterator();

    int value;
    int previous = -1;

    while (iter.has_next()) {
        value = iter.get_next();
        if (value != previous + 1) {
            printf("\tIncorrect value %d in list, that should be %d\n", value, previous + 1);
            return false;
        }
        previous = value;
    }

    if (value != NUMBER_OF_TESTS - 1) {
        printf("Apparently, some values were not found\n");
        return false;
    }

    return true;
}

__device__ void GPULinkedListTester::test_all()
{
    printf("GPU Linked List Tester\n");
    Tester::process_test(stress_test(), "Stress Test");
    print_summary();
}
