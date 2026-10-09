/**
 * Copyright (C) 2016, Daniel Thuerck
 * TU Darmstadt - Graphics, Capture and Massively Parallel Computing
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license. See the LICENSE file for details.
 */

#include <vector>
#include <queue>
#include <utility>
#include <map>
#include <set>
#include <iostream>
#include <numeric>
#include <algorithm>

#include "mapmap/compat/atomic.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"

#include "mapmap/header/graph.h"

NS_MAPMAP_BEGIN

/**
 * ****************************************************************************
 * ************ TBB task for lazy BFS-based component discovery ***************
 * ****************************************************************************
 */

/**
 * *****************************************************************************
 * ************************** Graph - public functions *************************
 * *****************************************************************************
 */

template<typename COSTTYPE>
FORCEINLINE
Graph<COSTTYPE>::
Graph(
    const luint_t num_nodes)
: m_num_nodes(num_nodes),
  m_nodes(num_nodes),
  m_edges(),
  m_components(num_nodes, 0),
  m_num_components(1),
  m_coloring(),
  m_was_colored(false)
{
    /* initial coloring: all 0 */
    m_coloring.resize(num_nodes);
    std::fill(m_coloring.begin(), m_coloring.end(), 0);
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
Graph<COSTTYPE>::
~Graph()
{

}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
add_edge(
    const luint_t node_a,
    const luint_t node_b,
    const COSTTYPE weight)
throw()
{
    if(std::max(node_a, node_b) >= m_num_nodes)
        throw std::runtime_error("Graph::add_edge: "
                "At least one ID larger than number of nodes in the graph.");

    m_edges.emplace_back(GraphEdge<COSTTYPE>{node_a, node_b, weight});

    const lint_t new_edge_id = m_edges.size() - 1;

    m_nodes[node_a].incident_edges.push_back(new_edge_id);
    m_nodes[node_b].incident_edges.push_back(new_edge_id);
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<GraphNode>&
Graph<COSTTYPE>::
nodes()
const
{
    return m_nodes;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const luint_t
Graph<COSTTYPE>::
num_nodes()
const
{
    return m_num_nodes;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<luint_t>&
Graph<COSTTYPE>::
inc_edges(
    const luint_t node)
const
{
    return m_nodes[node].incident_edges;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<GraphEdge<COSTTYPE>>&
Graph<COSTTYPE>::
edges()
const
{
    return m_edges;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const luint_t
Graph<COSTTYPE>::
num_edges()
const
{
    return m_edges.size();
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
update_components()
{
    m_num_components = 0;
    m_components.resize(m_nodes.size());

    std::vector<bool> visited(m_nodes.size(), false);
    std::vector<luint_t> queue;
    queue.reserve(m_nodes.size());

    for (luint_t start_node = 0; start_node < m_nodes.size(); ++start_node)
    {
        if (visited[start_node])
        {
            continue;
        }

        queue.clear();
        queue.push_back(start_node);
        visited[start_node] = true;

        for (size_t queue_pos = 0; queue_pos < queue.size(); ++queue_pos)
        {
            const luint_t current_node = queue[queue_pos];
            m_components[current_node] = m_num_components;

            for (const luint_t edge_id : inc_edges(current_node))
            {
                const GraphEdge<scalar_t<COSTTYPE>>& edge = m_edges[edge_id];
                const luint_t other_node = (edge.node_a == current_node) ?
                    edge.node_b : edge.node_a;

                if (!visited[other_node])
                {
                    visited[other_node] = true;
                    queue.push_back(other_node);
                }
            }
        }

        ++m_num_components;
    }
}
/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<luint_t>&
Graph<COSTTYPE>::
components()
const
{
    return m_components;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const luint_t
Graph<COSTTYPE>::
num_components()
const
{
    return m_num_components;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
set_coloring(
    const std::vector<luint_t>& coloring)
{
    std::copy(coloring.begin(),
        coloring.begin() + std::min((luint_t) coloring.size(), m_num_nodes),
        m_coloring.begin());
    m_was_colored = true;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
const std::vector<luint_t>&
Graph<COSTTYPE>::
get_coloring()
{
    return m_coloring;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
bool
Graph<COSTTYPE>::
was_colored()
{
    return m_was_colored;
}

/* ************************************************************************** */

template<typename COSTTYPE>
FORCEINLINE
void
Graph<COSTTYPE>::
sort_incidence_lists()
{
    /* sort incidence list after connected nodes */
    struct inc_sorter {
        inc_sorter(luint_t node, const GraphEdge<COSTTYPE> * edges) :
            m_node(node), m_edges(edges) {};
        ~inc_sorter() {};

        bool operator()(const luint_t e_id1, const luint_t e_id2)
        {
            const GraphEdge<COSTTYPE>& e1 = m_edges[e_id1];
            const luint_t o_node1 = (e1.node_a == m_node) ?
                e1.node_b : e1.node_a;

            const GraphEdge<COSTTYPE>& e2 = m_edges[e_id2];
            const luint_t o_node2 = (e2.node_a == m_node) ?
                e2.node_b : e2.node_a;

            return (o_node1 < o_node2);
        };

        const luint_t m_node;
        const GraphEdge<COSTTYPE> * m_edges;
    };

    tbb::blocked_range<luint_t> node_range(0, m_num_nodes);
    tbb::parallel_for(node_range,
        [&](const tbb::blocked_range<luint_t>& r)
        {
            for(luint_t n = r.begin(); n != r.end(); ++n)
            {
                std::sort(m_nodes[n].incident_edges.begin(),
                    m_nodes[n].incident_edges.end(),
                    inc_sorter(n, m_edges.data()));
            }
        });
}

NS_MAPMAP_END
