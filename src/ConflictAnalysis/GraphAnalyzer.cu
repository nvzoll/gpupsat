#include "GraphAnalyzer.cuh"

__device__ void get_conflicting_assignment(int vertex_index,
        Lit *conflicting_vertices_indices, int& next_position,
        CUDAListGraph& graph,
        Decision& largest_dec_level, Decision& second_largest);


__device__ void add_lit_to_learnt_clause(Lit *literals, int& next_position, Decision learnt,
        Decision& largest_dec_level, Decision& second_largest)
{
    literals[next_position] = learnt.literal;
    next_position++;


    learnt.literal = ~learnt.literal;

    if (learnt.decision_level > largest_dec_level.decision_level) {
        second_largest = largest_dec_level;
        largest_dec_level = learnt;
    }
    else {
        if (learnt.decision_level > second_largest.decision_level
            //&& learnt.decision_level != largest_dec_level.decision_level
           ) {
            second_largest = learnt;
        }
    }
}

__device__ Clause analyze_graph(CUDAListGraph& graph,
                                Decision& dec_level_to_backtrack,
                                Decision& highest)
{

    Decision largest_dec_level;
    largest_dec_level.decision_level = -1;
    Decision second_largest;
    second_largest.decision_level = -2;


#ifdef USE_ASSERTIONS
    assert(graph.are_all_unflagged());
    if (!graph.is_set(graph.get_conflict_vertex_index())) {
        printf("Attempting to analyzer conflict when no conflict is identified in graph!\n");
        assert(false);
    }
#endif

    Lit conflicting_vertices_indices[MAX_CONFLICTING_VERTICES_SIZE];


    int next_position = 0;

    /*
    CUDAListGraph::Iterator conflict_it = graph.get_conflict_vertex_back_iterator();

    while (conflict_it.has_next())
    {
        int next_index = conflict_it.get_next_index();
        get_conflicting_assignment(next_index,
                conflicting_vertices_indices, next_position, graph,
                largest_dec_level, second_largest);
    }
    */

    get_conflicting_assignment(graph.get_conflict_vertex_index(),
                               conflicting_vertices_indices, next_position, graph,
                               largest_dec_level, second_largest);

    Clause c;
    create_clause_on_dev(next_position, c);


    for (int i = 0; i < next_position; i++) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            //printf("Created clause, now adding adding literal ");
            //print_lit(~(conflicting_vertices_indices[i]));
            //printf(" of index %d\n", i);
        }
        addLitToDev(~(conflicting_vertices_indices[i]), c);
    }

    graph.unflag_all();

    if (next_position == 1) {
        dec_level_to_backtrack.decision_level = 0;
    }
    else {
        dec_level_to_backtrack = second_largest;
    }

    highest = largest_dec_level;

    return c;

}

__device__ void get_conflicting_assignment(int vertex_index,
        Lit *conflicting_vertices_indices, int& next_position, CUDAListGraph& graph,
        Decision& largest_dec_level, Decision& second_largest
                                          )
{

    if (graph.is_flagged(vertex_index)) {
        return;
    }
    else {
        graph.flag(vertex_index);
    }

    CUDAListGraph::Iterator it = graph.get_iterator(vertex_index, false);

    Decision current_decision = graph.get(vertex_index);

    if (!it.has_next()) {
#ifdef USE_ASSERTIONS
        assert(next_position < MAX_CONFLICTING_VERTICES_SIZE);
#endif

        //conflicting_vertices_indices[next_position] = current_decision.literal;
        //next_position++;

        add_lit_to_learnt_clause(conflicting_vertices_indices, next_position,
                                 current_decision, largest_dec_level, second_largest);

        return;
    }

    while (it.has_next()) {

        int next_vert = it.get_next_index();
        Decision next_decision = graph.get(next_vert);
        if (next_decision.decision_level < current_decision.decision_level) {
            if (!graph.is_flagged(next_vert)) {

                add_lit_to_learnt_clause(conflicting_vertices_indices,
                                         next_position, next_decision, largest_dec_level, second_largest);
                graph.flag(next_vert);
            }
        }
        else {
#ifdef USE_ASSERTIONS
            assert(next_decision.decision_level == current_decision.decision_level);
#endif
            get_conflicting_assignment(next_vert, conflicting_vertices_indices, next_position, graph,
                                       largest_dec_level, second_largest);
        }
    }

}
