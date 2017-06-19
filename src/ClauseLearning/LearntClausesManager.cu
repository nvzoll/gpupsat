#include "LearntClausesManager.cuh"

__device__ LearntClausesManager::LearntClausesManager(WatchedClausesList *watched_clauses) :
    repository(MAX_LEARNT_CLAUSES_PER_THREAD)
{
    if (watched_clauses != nullptr) {
        this->watched_clauses = watched_clauses;
    }
}

__device__ unsigned int LearntClausesManager::get_repository_capacity()
{
    return repository.get_capacity();
}

__device__ Clause *LearntClausesManager::learn_clause(Clause c)
{

    if (watched_clauses != nullptr && c.n_lits == 1) {
        return nullptr;
    }

    Clause removed;
    Clause *new_pointer;

    if (repository.add_clause(c, new_pointer, removed)) {
        if (watched_clauses != nullptr) {
            watched_clauses->replace_clause(removed, *new_pointer);
        }

        delete[] removed.literals;
    }
    else {
        if (watched_clauses != nullptr) {
            Lit dummy;
            watched_clauses->new_clause(new_pointer, dummy, true);
        }
    }

    return new_pointer;
}

__device__ void LearntClausesManager::set_watched_clauses(
    WatchedClausesList *watched_clauses)
{
    this->watched_clauses = watched_clauses;
}

__device__ sat_status LearntClausesManager::status_through_full_scan(
    VariablesStateHandler *handler, Clause *&conflicting_clause,
    GPULinkedList<found_implication>& implications)
{
    return repository.status_through_full_scan(handler, conflicting_clause, implications);

}

__device__ void LearntClausesManager::copy_clauses(GPULinkedList<Clause>& copied_clauses)
{
    repository.copy_clauses(copied_clauses);
}

__device__ void LearntClausesManager::print_learnt_clauses_repository()
{
    repository.print_structure();
}
