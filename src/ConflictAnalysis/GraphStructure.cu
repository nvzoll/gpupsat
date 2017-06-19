#include "CUDAListGraph.cuh"

__host__ __device__
CUDAListGraph::GraphStructure::GraphStructure(size_t vert_capacity, size_t ed_capacity)
    : vertices_indices{ vert_capacity + 1 }
    , vertices_capacity{ vert_capacity + 1 }
    , edges_capacity{ ed_capacity + 1 }
#ifdef __CUDA_ARCH__
    , vertices { new Vertex[vertices_capacity]}
, backward_edges{ new Edge[vertices_capacity * edges_capacity] }
#ifdef INCLUDE_FORWARD_EDGES
, edges { new Edge[vertices_capacity * edges_capacity] }
#endif
#endif
{
#ifndef __CUDA_ARCH__
    cudaMalloc(&(this->vertices), sizeof(Vertex) * vertices_capacity);

    size_t pitch;

#ifdef INCLUDE_FORWARD_EDGES
    cudaMallocPitch(&(this->edges), &pitch, sizeof(Edge) * vertices_capacity, sizeof(Edge) * edges_capacity);
#endif

    cudaMallocPitch(((void **)&backward_edges), (size_t *)&pitch, (size_t)vertices_capacity, sizeof(Edge) * edges_capacity);
#endif

    for (size_t i = 0; i < this->vertices_capacity; i++)
    {
        Vertex vertex;

        if (i < (vertices_capacity - 1)) {
            Var v = i;
            Lit l = mkLit(v, true);
            Decision d;
            d.decision_level = -30;
            d.literal = l;

            vertex.decision = d;
#ifdef INCLUDE_FORWARD_EDGES
            vertex.n_neighbors = 0;
#endif
            vertex.n_backward_neighbors = 0;
            vertex.set = false;
            vertex.flagged = false;
        }
        else {
            // Conflict vertex
            vertex.decision.decision_level = CONFLICT_VERTEX_DECISION_LEVEL;
            vertex.n_backward_neighbors = 0;
#ifdef INCLUDE_FORWARD_EDGES
            vertex.n_neighbors = 0;
#endif
            vertex.set = false;

        }
#ifndef __CUDA_ARCH__
        assert( cudaMemcpy(&(vertices[i]), &vertex, sizeof(Vertex), cudaMemcpyHostToDevice) == cudaSuccess );
#else
        memcpy(&(vertices[i]), &vertex, sizeof(Vertex));
#endif
    }
}

CUDAListGraph::GraphStructure::~GraphStructure()
{
    delete[] vertices;
    delete[] backward_edges;
#ifdef INCLUDE_FORWARD_EDGES
    delete[] edges;
#endif
}

__device__ bool CUDAListGraph::GraphStructure::contains(Decision decision)
{
    return vertices_indices.contains(var(decision.literal));
}

__device__ void CUDAListGraph::GraphStructure::add(Decision d)
{

    Var v = var(d.literal);
    int index = v;
#ifdef USE_ASSERTIONS
    assert(!vertices_indices.contains(index));
    assert(!vertices[index].set);
#endif

    vertices[index].decision = d;
#ifdef INCLUDE_FORWARD_EDGES
    vertices[index].n_neighbors = 0;
#endif
    vertices[index].n_backward_neighbors = 0;
    vertices[index].set = true;

    vertices_indices.add(index);
}

__device__ int CUDAListGraph::GraphStructure::get_conflict_vertex_index()
{
    return vertices_capacity - 1;
}

__device__ void CUDAListGraph::GraphStructure::add_edge(
    int src_vertex_index,
    int dest_vertex_index,
    const Clause *clause)
{

#ifdef USE_ASSERTIONS
    assert(src_vertex_index != dest_vertex_index);
    assert(!linked(src_vertex_index, dest_vertex_index));
    assert(vertices_indices.contains(src_vertex_index) && vertices[src_vertex_index].set);
    assert(vertices_indices.contains(dest_vertex_index) && vertices[dest_vertex_index].set);
#endif

    Vertex *dest_vertex = &(vertices[dest_vertex_index]);
#ifdef DEBUG
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Adding vertex %d to %d\n", src_vertex_index, dest_vertex_index);
    }

    Vertex *src_vertex = &(vertices[src_vertex_index]);

    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Source vertex = ");
        print_lit(src_vertex->decision.literal);
#ifdef INCLUDE_FORWARD_EDGES
        printf(" it has %d edges, dest vertex = ", src_vertex->n_neighbors);
        print_lit(dest_vertex->decision.literal);
#else // INCLUDE_FORWARD_EDGES
        printf(" ,dest vertex = ");
        print_lit(dest_vertex->decision.literal);
#endif // INCLUDE_FORWARD_EDGES

        printf(" it has %d backward edges\n", dest_vertex->n_backward_neighbors);
    }
#endif // DEBUG

#ifdef INCLUDE_FORWARD_EDGES
#ifdef USE_ASSERTIONS
    assert(src_vertex->n_neighbors < edges_capacity
           && dest_vertex->n_backward_neighbors < edges_capacity
          );
#endif
#else

#ifdef USE_ASSERTIONS
    assert(dest_vertex->n_backward_neighbors < edges_capacity
          );
#endif
#endif

#ifdef INCLUDE_FORWARD_EDGES
    Edge forward_edge;
    forward_edge.neighbor_index = dest_vertex_index;
    forward_edge.implicating_clause = clause;
    edges[src_vertex_index * edges_capacity + src_vertex->n_neighbors] = forward_edge;
    src_vertex->n_neighbors++;
#endif

    Edge backward_edge;
    backward_edge.neighbor_index = src_vertex_index;
    backward_edge.implicating_clause = clause;
    backward_edges[dest_vertex_index * edges_capacity + dest_vertex->n_backward_neighbors] = backward_edge;
    dest_vertex->n_backward_neighbors++;

#ifdef DEBUG
    if (DEBUG_SHOULD_PRINT(threadIdx.x, blockIdx.x)) {
        printf("Adding is done!\nSource vertex = ");
        print_lit(src_vertex->decision.literal);
#ifdef INCLUDE_FORWARD_EDGES
        printf(" it has %d edges", src_vertex->n_neighbors);
#endif
        printf(", dest vertex = ");
        print_lit(dest_vertex->decision.literal);
        printf(" it has %d backward edges\n", dest_vertex->n_backward_neighbors);


    }
#endif


}

__device__ bool CUDAListGraph::GraphStructure::linked_with_conflict(Var src)
{
    return linked(src, get_conflict_vertex_index());
}

__device__ bool CUDAListGraph::GraphStructure::linked(Var src, Var dest)
{
    for (int i = 0; i < vertices_indices.size_of(); i++) {
        int index = vertices_indices.get(i);
        Vertex v = vertices[index];

        if (var(v.decision.literal) == dest) {

            for (int j = 0; j < v.n_backward_neighbors; j++) {

                Edge *e = get_edge(index, j, false);

                if (var(vertices[e->neighbor_index].decision.literal) == src) {
                    return true;
                }
            }
            return false;
        }

    }
    return false;
}

__device__ void CUDAListGraph::GraphStructure::remove_vertex(int vertex_index)
{

    Vertex *v = &(vertices[vertex_index]);
    v->set = false;
    v->flagged = false;
    v->n_backward_neighbors = 0;
#ifdef INCLUDE_FORWARD_EDGES
    v->n_neighbors = 0;
#endif

    vertices_indices.remove_obj(vertex_index);

    for (int i = 0; i < vertices_indices.size_of(); i++) {
        int other_index = vertices_indices.get(i);
        Vertex *other = &(vertices[other_index]);

#ifdef INCLUDE_FORWARD_EDGES
        // Forward edges
        for (int j = 0; j < other->n_neighbors; j++) {
            Edge *forward = get_edge(other_index, j, true);
            if (forward->neighbor_index == vertex_index) {
                remove_edge(other_index, j, true);
                break;
            }
        }
#endif

        // Backward edges
        for (int j = 0; j < other->n_backward_neighbors; j++) {
            Edge *backward = get_edge(other_index, j, false);
            if (backward->neighbor_index == vertex_index) {
                remove_edge(other_index, j, false);
                break;
            }
        }

    }

}



__device__ Edge *CUDAListGraph::GraphStructure::get_edge(int src_vertex_index, int edge_index, bool forward)
{
    if (forward) {
#ifdef INCLUDE_FORWARD_EDGES
        return &(edges[src_vertex_index * edges_capacity + edge_index]);
#else
        assert(false);
#endif
    }
    else {
        return &(backward_edges[src_vertex_index * edges_capacity + edge_index]);
    }

    return 0;
}

__device__ void CUDAListGraph::GraphStructure::remove_edge(int vertex_index, int edge_index, bool forward)
{

    for (int i = edge_index; i < edges_capacity - 1; i++) {
        if (forward) {
#ifdef INCLUDE_FORWARD_EDGES
            *get_edge(vertex_index, i, true) = *get_edge(vertex_index, i + 1, true);
#else
            assert(false);
#endif
        }
        else {
            *get_edge(vertex_index, i, false) = *get_edge(vertex_index, i + 1, false);
        }
    }

    if (forward) {
#ifdef INCLUDE_FORWARD_EDGES
        vertices[vertex_index].n_neighbors--;
#else
        assert(false);
#endif
    }
    else {
        vertices[vertex_index].n_backward_neighbors--;
    }
}

__device__ void CUDAListGraph::GraphStructure::reset()
{

    /*

    while (!vertices_indices.empty())
    {
        int current_index = vertices_indices.get(0);
        remove_vertex(current_index);
    }

    vertices_indices.clear();
    */


    for (int i = vertices_indices.size_of() - 1; i >= 0; i--) {
        int current_index = vertices_indices.get(i);

        if (!vertices[current_index].decision.implicated_from_formula) {
            remove_vertex(current_index);
        }
    }


}
// Conflict edge
__device__ void CUDAListGraph::GraphStructure::link_with_conflict(
    int src_vertex_index,
    const Clause *implicating_clause,
    int current_decision_level)
{
    int conflict_vertex_index = get_conflict_vertex_index();

    if (!vertices[conflict_vertex_index].set) {
        vertices[conflict_vertex_index].set = true;
        vertices_indices.add(conflict_vertex_index);
    }

#ifdef USE_ASSERTIONS
    assert(vertices_indices.contains(conflict_vertex_index) &&
           vertices[conflict_vertex_index].set);
#endif



#ifdef USE_ASSERTIONS
    if (linked(src_vertex_index, get_conflict_vertex_index())) {
        printf("Attempting to link two vertices (%d, k) that are already linked.\n",
               src_vertex_index);
        //print();
        assert(false);
    }
#endif


    vertices[conflict_vertex_index].decision.decision_level = current_decision_level;
    vertices[conflict_vertex_index].decision.implicated_from_formula = false;

    add_edge(src_vertex_index, vertices_capacity - 1, implicating_clause);

#ifdef USE_ASSERTIONS
    assert(vertices_indices.contains(conflict_vertex_index) &&
           vertices[conflict_vertex_index].set);
#endif

}


// For test
__device__ bool CUDAListGraph::GraphStructure::check_consistency()
{
    for (int i = 0; i < vertices_capacity; i++) {
        Vertex *v = &(vertices[i]);

        if (v->set == true) {
            if (!vertices_indices.contains(i)) {
                printf("Vertices indices does not contain the set vertex %d.\n", i);
                return false;
            }

            if (
#ifdef INCLUDE_FORWARD_EDGES
                v->n_neighbors < 0 || v->n_neighbors > edges_capacity ||
#endif
                v-> n_backward_neighbors < 0 ||
                v->n_backward_neighbors > edges_capacity
            ) {
#ifdef INCLUDE_FORWARD_EDGES
                printf("Number of neighbors (%d) or backward neighbors (%d) of vertex of index %d is invalid\n", v->n_neighbors,
                       v->n_backward_neighbors, i);
#else
                printf("Number of backward neighbors (%d) of vertex of index %d is invalid\n",
                       v->n_backward_neighbors, i);
#endif
                return false;
            }

#ifdef INCLUDE_FORWARD_EDGES
            for (int k = 0; k < v->n_neighbors; k++) {
                Edge *e = get_edge(i, k, true);
                Clause *c = e->implicating_clause;

                if (e->neighbor_index == i) {
                    printf("Vertex %d has a forward edge to itself\n", i);
                    return false;
                }

                if (c->n_lits > c->capacity) {
                    printf("One clause in forward edge of vertex %d is invalid.\n", i);
                    return false;
                }

                Vertex *dest = &(vertices[e->neighbor_index]);

                bool has_backward = false;
                for (int j = 0; j < dest->n_backward_neighbors; j++) {
                    Edge *backward = get_edge(e->neighbor_index, j, false);

                    if (backward->neighbor_index == i) {
                        has_backward = true;
                        Clause *clause2 = backward->implicating_clause;

                        if (c->capacity != clause2->capacity
                            || c->literals != clause2->literals
                            || c->n_lits != clause2->n_lits
                           ) {
                            printf("Forward and backward edges do not have same clause!\n");
                            return false;
                        }


                    }
                }

                if (!has_backward) {
                    printf("Edge of vertex %d does not have a correspondent backward\n", i);
                    return false;
                }

                bool has_var = false;
                for (int l = 0; l < c->n_lits; l++) {
                    if (var(c->literals[l]) == var(v->decision.literal)) {
                        has_var = true;
                    }
                }

                if (!has_var) {
                    printf("Forward edge does not have its var in its clause\n");
                    return false;
                }


            }
#endif
            for (int k = 0; k < v->n_backward_neighbors; k++) {
                Edge *e = get_edge(i, k, false);

                for (int l = k + 1; l < v->n_backward_neighbors; l++) {
                    Edge *e2 = get_edge(i, l, false);
                    if (e2->neighbor_index == e->neighbor_index) {
                        printf("Vertex ");
                        print_vertex(i);
                        printf(" has two (or more) edges to vertex ");
                        print_vertex(e->neighbor_index);
                        printf("\n");
                        return false;
                    }
                }

                if (e->neighbor_index == i) {
                    printf("Vertex %d has a backward edge to itself\n", i);
                    return false;
                }

            }

        }
        else {
            if (vertices_indices.contains(i)) {
                printf("Vertices indices contains the not set vertex %d.\n", i);
                return false;
            }
        }

    }

    return true;
}


__device__ void CUDAListGraph::GraphStructure::print()
{
    //for (int i = 0; i < vertices_capacity; i++)
    for (int j = 0; j < vertices_indices.size_of(); j++) {
        int i = vertices_indices.get(j);

        if (vertices[i].set) {
            printf("Vertex:");
            if (i != vertices_capacity - 1) {
                print_decision(vertices[i].decision);
            }
            else {
                printf("k");
                printf("(%d)", vertices[i].decision.decision_level);
            }
            printf("\n");

            if (
#ifdef INCLUDE_FORWARD_EDGES
                vertices[i].n_neighbors > 0 ||
#endif
                vertices[i].n_backward_neighbors > 0

            ) {
                printf("\tNeighbors:\n");
            }
        }
#ifdef INCLUDE_FORWARD_EDGES
        if (vertices[i].set && vertices[i].n_neighbors > 0) {
            printf("\t\tForward: ");

            for (int j = 0; j < vertices[i].n_neighbors; j++) {
                if (edges[i * edges_capacity + j].neighbor_index != vertices_capacity - 1) {
                    printf("%d (", edges[i * edges_capacity + j].neighbor_index);
                }
                else {
                    printf("k (");
                }
                print_clause(*(edges[i * edges_capacity + j].implicating_clause));
                printf(") ");
            }

            printf("\n");

        }
#endif

        if (vertices[i].set && vertices[i].n_backward_neighbors > 0) {
            printf("\t\tBackward: ");
            for (int j = 0; j < vertices[i].n_backward_neighbors; j++) {
                if (backward_edges[i * edges_capacity + j].neighbor_index != vertices_capacity - 1) {
                    printf("%d (", backward_edges[i * edges_capacity + j].neighbor_index);
                }
                else {
                    printf("k (");
                }
                print_clause(*(backward_edges[i * edges_capacity + j].implicating_clause));
                printf(") ");
            }

            printf("\n");

        }



    }
}

__device__ void CUDAListGraph::GraphStructure::print_vertex(Var var)
{
    //if (vertices[var].set)
    //{
    printf("Vertex:");
    if (var != vertices_capacity - 1) {
        print_decision(vertices[var].decision);
    }
    else {
        printf("k");
    }
    printf(" (%d)", vertices[var].decision.decision_level);
    printf(", set = %s\n", vertices[var].set ? "T" : "F");

    if (
#ifdef INCLUDE_FORWARD_EDGES
        vertices[var].n_neighbors > 0 ||
#endif
        vertices[var].n_backward_neighbors > 0

    ) {
        printf("\tNeighbors:\n");
    }
    //}
#ifdef INCLUDE_FORWARD_EDGES
    if (vertices[var].set && vertices[var].n_neighbors > 0) {
        printf("\t\tForward: ");
        for (int j = 0; j < vertices[var].n_neighbors; j++) {
            if (edges[var * edges_capacity + j].neighbor_index != vertices_capacity - 1) {
                printf("%d (", edges[var * edges_capacity + j].neighbor_index);
            }
            else {
                printf("k (");
            }
            print_clause(*(edges[var * edges_capacity + j].implicating_clause));
            printf(") ");
        }

        printf("\n");

    }
#endif

    if (vertices[var].set && vertices[var].n_backward_neighbors > 0) {
        printf("\t\tBackward: ");
        for (int j = 0; j < vertices[var].n_backward_neighbors; j++) {
            if (backward_edges[var * edges_capacity + j].neighbor_index != vertices_capacity - 1) {
                printf("%d (", backward_edges[var * edges_capacity + j].neighbor_index);
            }
            else {
                printf("k (");
            }
            print_clause(*(backward_edges[var * edges_capacity + j].implicating_clause));
            printf(") ");
        }

        printf("\n");

    }

}
__device__ void CUDAListGraph::GraphStructure::print_conflict_vertex()
{
    print_vertex(vertices_capacity - 1);
}
