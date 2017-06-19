#include "LearntClauseRepository.cuh"

__device__ LearntClauseRepository::LearntClauseRepository(int repository_size) :
    //clauses(repository_size)
    clauses()
{
    this->position_to_switch = 0;
}

__device__ unsigned int LearntClauseRepository::get_capacity()
{
    return clauses.get_capacity();
}

__device__ bool LearntClauseRepository::add_clause(Clause learnt_clause,
        Clause *&allocation_pointer, Clause& removed)
{
    if (!clauses.full()) {
        clauses.add(learnt_clause);
        allocation_pointer = clauses.get_ptr(clauses.size_of() - 1);
        return false;
    }
    else {
        removed = clauses.reset(learnt_clause, position_to_switch);
        allocation_pointer = clauses.get_ptr(position_to_switch);
        position_to_switch++;

        if (position_to_switch >= clauses.size_of()) {
            position_to_switch = 0;
        }

        return true;
    }
}
__device__ sat_status LearntClauseRepository::status_through_full_scan(
    VariablesStateHandler *handler, Clause *&conflicting_clause,
    GPULinkedList<found_implication>& implications
)
{
    conflicting_clause = nullptr;
    bool has_undef = false;
    for (int i = 0; i < clauses.size_of(); i++) {
        Clause c = clauses.get(i);
        Lit learnt;
        sat_status stat = handler->clause_status(c, &learnt);

        if (learnt.x != -1) {
            found_implication fi;
            Decision d;
            d.decision_level = handler->get_decision_level();
            d.literal = learnt;
            fi.implicating_clause = clauses.get_ptr(i);
            fi.implication = d;
            implications.push_back(fi);
        }

        if (stat == sat_status::UNSAT) {
            conflicting_clause = clauses.get_ptr(i);
            return sat_status::UNSAT;
        }

        if (stat == sat_status::UNDEF) {
            has_undef = true;
        }

    }

    return has_undef ? sat_status::UNDEF : sat_status::SAT;
}

__device__ void LearntClauseRepository::copy_clauses(GPULinkedList<Clause>& copied_clauses)
{
    for (int i = 0; i < clauses.size_of(); i++) {
        copied_clauses.push_back(clauses.get(i));
    }
}


__device__ void LearntClauseRepository::print_structure()
{
    printf("Learnt clauses repository:\n");
    printf("Clauses: \n");
    for (int i = 0; i < clauses.size_of(); i++) {
        if (i == position_to_switch) {
            printf("==>");
        }
        else {
            printf("   ");
        }
        print_clause(clauses.get(i));
        printf("\n");
    }

}
