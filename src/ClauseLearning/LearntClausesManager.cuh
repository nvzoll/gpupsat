#ifndef __LEARNTCLAUSESMANAGER_CUH__
#define __LEARNTCLAUSESMANAGER_CUH__

#include "LearntClauseRepository.cuh"
#include "BCPStrategy/WatchedClausesList.cuh"
#include "SATSolver/SolverTypes.cuh"

class LearntClausesManager
{
private:
    LearntClauseRepository repository;
    WatchedClausesList *watched_clauses;
public:
    __device__ LearntClausesManager(WatchedClausesList *watched_clauses);
    __device__ Clause *learn_clause(Clause c);
    __device__ void set_watched_clauses(WatchedClausesList *watched_clauses);
    __device__ sat_status status_through_full_scan(VariablesStateHandler *handler,
            Clause *&conflicting_clause,
            GPULinkedList<found_implication>& implications);
    __device__ void copy_clauses(GPULinkedList<Clause>& copied_clauses);
    __device__ unsigned int get_repository_capacity();

    /**
     * Test methods
     */
    __device__ void print_learnt_clauses_repository();
};

#endif /* __LEARNTCLAUSESMANAGER_CUH__ */
