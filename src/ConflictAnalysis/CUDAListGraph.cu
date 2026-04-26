#include "CUDAListGraph.cuh"

__host__ __device__ CUDAListGraph::CUDAListGraph(int vertices_capacity, int ed_capacity):
    structure(vertices_capacity, ed_capacity)
{

}

__device__ void CUDAListGraph::set(Decision d)
{
    structure.add(d);

}

__device__ bool CUDAListGraph::is_set(Var var)
{
    return structure.vertices[var].set;
}

__device__ Decision CUDAListGraph::get(int index)
{
    return structure.vertices[index].decision;
}

__device__ int CUDAListGraph::get_n_neighbors(int vertex_index, bool forward)
{
    if (forward) {
#ifdef INCLUDE_FORWARD_EDGES
        return structure.vertices[vertex_index].n_neighbors;
#else
        assert(false);
#endif
    }
    else {
        return structure.vertices[vertex_index].n_backward_neighbors;
    }

    return 0;
}

__device__ int CUDAListGraph::get_neighbor_index(int vertex_index, int edge_index,
        bool forward)
{
    return structure.get_edge(vertex_index, edge_index, forward)->neighbor_index;
}
__device__ int CUDAListGraph::get_conflict_vertex_index()
{
    return structure.get_conflict_vertex_index();
}

__device__ bool CUDAListGraph::linked(Var src, Var dest)
{
    return structure.linked(src, dest);
}

__device__ bool CUDAListGraph::linked_with_conflict(Var src)
{
    return structure.linked_with_conflict(src);
}

__device__ void CUDAListGraph::link(Var src, Var dest, const Clause *implicating_clause)
{
    int src_index = src;
    int dest_index = dest;
#ifdef INCLUDE_FORWARD_EDGES
#endif


    structure.add_edge(src_index, dest_index, implicating_clause);
}

__device__ void CUDAListGraph::link_with_conflict(int src_vertex_index,
        const Clause *implicating_clause,
        int current_decision_level)
{
    structure.link_with_conflict(src_vertex_index,
                                 implicating_clause, current_decision_level);
}

__device__ void CUDAListGraph::backtrack_to(int decision_level)
{
    for (int i = 0; i < structure.vertices_indices.size_of(); i++) {
        int vertex_index = structure.vertices_indices.get(i);
        Vertex *v = &(structure.vertices[vertex_index]);

        // TODO greater than or greater or equal to
        if (v->decision.decision_level > decision_level
            && !v->decision.implicated_from_formula) {
            structure.remove_vertex(vertex_index);
            i--;
        }


    }

}
__device__ bool CUDAListGraph::contains(Var var)
{
    Decision pos;
    Decision neg;

    pos.literal = mkLit(var, true);
    neg.literal = mkLit(var, false);

    return structure.contains(pos) || structure.contains(neg);

}

__device__ void CUDAListGraph::print()
{
    structure.print();
}

__device__ bool CUDAListGraph::check_consistency()
{
    return structure.check_consistency();
}


__device__ void CUDAListGraph::reset()
{
    structure.reset();
}



__device__ CUDAListGraph::Iterator::Iterator(bool forward, int vertex_index, CUDAListGraph *graph)
{
    this->forward = forward;
    this->vertex_index = vertex_index;
    this->next = 0;
    this->graph = graph;
}

__device__ bool CUDAListGraph::Iterator::has_next()
{
    return next < graph->get_n_neighbors(vertex_index, forward);
}

__device__ int CUDAListGraph::Iterator::get_next_index()
{
    next++;
    return graph->get_neighbor_index(vertex_index, next - 1, forward);
}

__device__ bool CUDAListGraph::Iterator::is_forward()
{
    return forward;
}


// Iterator
__device__ CUDAListGraph::Iterator
CUDAListGraph::get_iterator(int vertex_index, bool forward)
{
    Iterator iter(forward, vertex_index, this);
    return iter;
}
__device__ CUDAListGraph::Iterator CUDAListGraph::get_conflict_vertex_back_iterator()
{
    return get_iterator(structure.get_conflict_vertex_index(), false);
}

__device__ void CUDAListGraph::flag(int vertex_index)
{
    Vertex *vertex = &(structure.vertices[vertex_index]);

    vertex->flagged = true;

}
__device__ void CUDAListGraph::unflag(int vertex_index)
{
    Vertex *vertex = &(structure.vertices[vertex_index]);
    vertex->flagged = false;
}
__device__ void CUDAListGraph::unflag_all()
{
    for (int i = 0; i < structure.vertices_indices.size_of(); i++) {
        unflag(structure.vertices_indices.get(i));
    }
}
__device__ bool CUDAListGraph::is_flagged(int vertex_index)
{
    return structure.vertices[vertex_index].flagged;
}

__device__ bool CUDAListGraph::are_all_unflagged()
{
    for (int i = 0; i < structure.vertices_capacity; i++) {
        if (is_flagged(i)) {
            return false;
        }
    }

    return true;
}

__device__ void CUDAListGraph::print_vertex(Var var)
{
    structure.print_vertex(var);
}
__device__ void CUDAListGraph::print_conflict_vertex()

{
    structure.print_conflict_vertex();
}
