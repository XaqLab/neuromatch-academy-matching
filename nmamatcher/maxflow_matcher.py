# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:45:42 2020

@author: Zhe
"""

import numpy as np


class FlowGraph:
    r"""Base class for a flow graph.

    Args
    ----
    vertices: list
        The vertices in the flow graph, in which each element is immutable
        objects, e.g. str, int or tuple.
    cost: (N, N), sparse matrix
        The cost matrix with `cost[i, j]` as the cost on edge `i` --> `j`.
    capacity: (N, N), sparse matrix
        The capacity matrix with `capacity[i, j]` as the capacity on edge `i`
        ---> `j`.

    """
    def __init__(self, vertices, cost, capacity):
        assert set(['source', 'sink']).issubset(set(vertices))
        N = len(set(vertices))
        assert cost.shape==(N, N) and capacity.shape==(N, N)

        self.vertices = vertices
        self.cost, self.capacity = cost, capacity
        self.residual = capacity.copy() # the residual flow graph
        self.max_flow = 0.

    def get_augmenting_path(self, random_order=False):
        r"""Returns the shortest path from source to sink on residual flow
        graph.

        Bell-Fordman algorithm with BFS are used. Starting from the source,
        new vertices are visited and relaxed. The shortest path to the sink
        will be the next augmenting path if it exists.

        Args
        ----
        random_order: bool
            New vertices are pushed into the queue with random order if set to
            ``True``.

        Returns
        -------
        parent: (N,), array_like
            The parent of each vertices along the shortest path to the source.

        """
        d = np.empty((len(self.vertices),), dtype=np.float) # distance to source
        d.fill(np.inf)
        parent = np.empty((len(self.vertices),), dtype=np.int) # parent index along the shortest path
        parent.fill(-1)

        queue = [self.vertices.index('source')]
        d[queue[0]] = 0
        while queue:
            u = queue.pop(0)
            _, vs = (self.residual[u]>0).nonzero()
            if random_order:
                vs = np.random.permutation(vs)
            for v in vs:
                if d[u]+self.cost[u, v]<d[v]: # relax edge
                    d[v] = d[u]+self.cost[u, v]
                    parent[v] = u
                    if v not in queue:
                        queue.append(v)
        return parent

    def FordFulkerson(self):
        r"""Implements Ford-Fulkerson algorithm to find min-cost max-flow.

        Augmenting paths are repeatedly calculated and used to update the
        residual flow graph. Algorithm terminates when no augmenting path can
        be found

        """
        while True:
            parent = self.get_augmenting_path(True)
            if parent[self.vertices.index('sink')]<0:
                # source is unconnected to sink on the residual flow graph
                break
            self.print_path(parent)

            # calculate flow increase
            delta_flow = np.inf
            v = self.vertices.index('sink')
            while self.vertices[v]!='source':
                u = parent[v]
                delta_flow = min(delta_flow, self.residual[u, v])
                v = u
            self.max_flow += delta_flow

            # update residual
            v = self.vertices.index('sink')
            while self.vertices[v]!='source':
                u = parent[v]
                self.residual[u, v] -= delta_flow # forward arc
                self.residual[v, u] += delta_flow # backward arc
                v = u

    def print_path(self, parent):
        r"""Prints a path from source to sink.

        Args
        ----
        parent: (N,), array_like
            The parent of each vertices along the shortest path to the source.

        """
        idxs = [self.vertices.index('sink')]
        while self.vertices[idxs[-1]]!='source':
            idxs.append(parent[idxs[-1]])
        idxs.reverse()
        print('-->'.join([str(self.vertices[i]) for i in idxs]))
