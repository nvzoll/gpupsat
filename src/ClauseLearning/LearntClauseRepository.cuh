#ifndef __LEARNTCLAUSEREPOSITORY_CUH__
#define __LEARNTCLAUSEREPOSITORY_CUH__

#include "Utils/GPUVec.cuh"
#include "Utils/GPUStaticVec.cuh"
#include "SATSolver/SolverTypes.cuh"
#include "SATSolver/Configs.cuh"
#include "Utils/GPULinkedList.cuh"
#include "SATSolver/VariablesStateHandler.cuh"

class LearntClauseRepository
{
private:
    //GPUVec<Clause> clauses;
    GPUStaticVec<Clause, MAX_LEARNT_CLAUSES_PER_THREAD> clauses;
    int position_to_switch;
public:
    __device__ LearntClauseRepository(int repository_size);
    /**
     * Adds a clause to the repository, if there is no more space, one is removed
     * (using some strategy). It receives the learnt clause and returns a pointer
     * to where it is now stored, the clause that was removed (if any) and
     * as the return value, it returns if a clause was removed or not.
     * The parameter removed_clause will only contains a valid clause after the call
     * of this method if it returns true.
     */
    __device__ bool add_clause(Clause learnt_clause, Clause *&allocation_pointer, Clause& removed);

    __device__ sat_status status_through_full_scan(VariablesStateHandler *handler,
            Clause *&conflicting_clause,
            GPULinkedList<found_implication>& implications);

    __device__ void copy_clauses(GPULinkedList<Clause>& copied_clauses);
    __device__ unsigned int get_capacity();

    /**
     * Test methods
     */
    __device__ void print_structure();
};


#endif /* __LEARNTCLAUSEREPOSITORY_CUH__ */
