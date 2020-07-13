# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:45:42 2020

@author: Zhe
"""

import numpy as np
from scipy.sparse import dok_matrix

from .utils import slot_label, preprocess

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances


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


class PodMentorGraph(FlowGraph):
    r"""Flow graph for pod-mentor matching.

    During week 1 of NMA, each pod is assigned with a mentor for project
    initialization.

    Args
    ----
    pod_info: dict
        The dictionary containing information of all pods returned by
        `.utils.load_pod_info`.
    mentor_info: dict
        The dictionary containing available time of all mentors returned by
        `.utils.load_mentor_info`.
    max_pod_per_mentor: int
        The maximum number of pods assigned to a mentor.
    d_idxs: list of int
        The days for project initialization, as indices in :math:`[0, 15)`.
        Default value ``[2, 3]`` means Wednesday and Thursday of the first
        week.
    use_second: bool
        Secondary choices from mentors are enabled. Currently, no choice
        preference is implemented, so that second choices are equivalent to
        first choices if enabled.
    affinity: (pod_num, mentor_num), array_like
        The content topic alignment matrix between pods and mentors.

    """
    def __init__(self, pod_info, mentor_info, max_pod_per_mentor=2,
                 d_idxs=[2, 3], use_second=False, affinity=None):
        self.pod_info = pod_info
        self.mentor_info = mentor_info
        pod_num, mentor_num = pod_info['pod_num'], mentor_info['mentor_num']
        if affinity is None:
            affinity = -np.ones((pod_num, mentor_num))

        # gather available slots
        mentor_slots = []
        for m_idx in range(mentor_num):
            day_slots, day_slots_ = [], []
            for d_idx in d_idxs:
                if d_idx in mentor_info['primary_days'][m_idx]:
                    for s_idx in mentor_info['primary_slots'][m_idx]:
                        day_slots_.append((d_idx, s_idx))
                        day_slots.append((d_idx, s_idx, 1.))
                if use_second and d_idx in mentor_info['secondary_days'][m_idx]:
                    for s_idx in mentor_info['secondary_slots'][m_idx]:
                        if (d_idx, s_idx) not in day_slots_: # first choice has higher priority
                            day_slots_.append((d_idx, s_idx))
                            day_slots.append((d_idx, s_idx, mentor_info['flexibility'][m_idx]))
            mentor_slots.append(day_slots)

        # define vertices in the graph
        vertices = ['source', 'sink']
        for p_idx in range(pod_num):
            vertices.append(('pod', p_idx))
        for m_idx in range(mentor_num):
            for d_idx, s_idx, _ in mentor_slots[m_idx]:
                vertices.append(('mentor', m_idx, d_idx, s_idx))
                if ('mentor', m_idx, d_idx) not in vertices:
                    vertices.append(('mentor', m_idx, d_idx))
            vertices.append(('mentor', m_idx))

        # define edge cost and capacity in sparse matrices
        cost = dok_matrix((len(vertices), len(vertices)))
        capacity = dok_matrix((len(vertices), len(vertices)))
        # source --> pod
        u = vertices.index('source')
        for p_idx in range(pod_num):
            v = vertices.index(('pod', p_idx))
            capacity[u, v] = 1
        # pod --> mentor-day-slot
        for p_idx in range(pod_num):
            for m_idx in range(mentor_num):
                for d_idx, s_idx, _ in mentor_slots[m_idx]:
                    if s_idx in pod_info['slots'][p_idx]%48: # assuming each pod is available for all week
                        u = vertices.index(('pod', p_idx))
                        v = vertices.index(('mentor', m_idx, d_idx, s_idx))
                        cost[u, v] = -int(100*affinity[p_idx, m_idx])
                        cost[v, u] = -cost[u, v]
                        capacity[u, v] = 1
        # mentor-day-slot --> mentor-day
        for m_idx in range(mentor_num):
            for d_idx, s_idx, f in mentor_slots[m_idx]:
                u = vertices.index(('mentor', m_idx, d_idx, s_idx))
                v = vertices.index(('mentor', m_idx, d_idx))
                cost[u, v] = -int(10*f)
                cost[v, u] = -cost[u, v]
                capacity[u, v] = 1
        # mentor-day-->mentor
        for m_idx in range(mentor_num):
            v = vertices.index(('mentor', m_idx))
            for d_idx in set([d_idx_ for d_idx_, _, _ in mentor_slots[m_idx]]):
                u = vertices.index(('mentor', m_idx, d_idx))
                capacity[u, v] = len([s_idx_ for d_idx_, s_idx_, _ in mentor_slots[m_idx] if d_idx_==d_idx])
        # mentor-->sink
        v = vertices.index('sink')
        for m_idx in range(mentor_num):
            u = vertices.index(('mentor', m_idx))
            capacity[u, v] = max_pod_per_mentor

        super(PodMentorGraph, self).__init__(vertices, cost, capacity)

    def get_matches(self):
        r"""Returns pod-mentor matches.

        Returns
        -------
        matches: list
            The matching information for each pod. Each element is a tuple like
            `(m_idx, d_idx, s_idx)` containing mentor index, day index and slot
            index. If a pod is not assigned with any mentor, the list element
            is ``None``.

        """
        self.FordFulkerson()
        flow = (self.capacity-self.residual).toarray()
        matches = []
        matched_count = 0
        for p_idx in range(self.pod_info['pod_num']):
            u = self.vertices.index(('pod', p_idx))
            vs, = (flow[u]>0).nonzero()
            if vs.size==1:
                _, m_idx, d_idx, s_idx = self.vertices[vs[0]]
                matches.append((m_idx, d_idx, s_idx))
                matched_count += 1
            elif vs.size==0:
                matches.append(None)
                print('{} not assigned with any mentor'.format(
                    self.pod_info['name'][p_idx]
                    ))
            else:
                raise RuntimeError(f'more than one outward flow detected for pod {p_idx}')
        print('{}/{} pods assigned with a mentor'.format(matched_count, len(matches)))
        return matches

    def export_mentor_schedule(self, out_csv):
        r"""Exports mentor schedule CSV file.

        Args
        ----
        out_csv: str
            The output CSV file.

        """
        matches = self.get_matches()

        available_count, assigned_count, usage_count = 0, 0, 0
        with open(out_csv, 'w') as f:
            f.write('name,email,day,slot (UNIVERSAL),mentor time zone,slot (LOCAL),pod,pod time zone group,zoom link\n')
            for m_idx in range(self.mentor_info['mentor_num']):
                m_name = ' '.join([
                    self.mentor_info['first_name'][m_idx],
                    self.mentor_info['last_name'][m_idx]
                    ])
                m_email = self.mentor_info['email'][m_idx]

                m_capacity = self.capacity[:, self.vertices.index(('mentor', m_idx))].toarray().sum()
                if m_capacity>0:
                    available_count += 1

                    assignments = []
                    for p_idx, match in enumerate(matches): # (m_idx, day, s_idx)
                        if match and match[0]==m_idx:
                            assignments.append((p_idx, match[1], match[2]))

                    if assignments:
                        assigned_count += 1
                        usage_count += len(assignments)/m_capacity

                        out_strs = []
                        for p_idx, d_idx, s_idx in assignments:
                            # convert from UTC to UTC+1
                            s_idx += 2
                            if s_idx>=48:
                                d_idx += 1
                                s_idx -= 48

                            if d_idx==2:
                                day = 'Wednesday'
                            elif d_idx==3:
                                day = 'Thursday'
                            elif d_idx==4:
                                day = 'Friday'

                            slot_universal = slot_label(s_idx)

                            m_tz = self.mentor_info['timezone'][m_idx]

                            # convert to local time zone
                            s_idx -= 2
                            if m_tz[3]=='+':
                                s_idx += int(m_tz[4:])*2
                            else:
                                s_idx -= int(m_tz[4:])*2
                            slot_local = slot_label(s_idx)

                            p_name = self.pod_info['name'][p_idx]
                            p_group = self.pod_info['timezone_label'][p_idx]

                            z_link = ''

                            out_strs.append(','.join([
                                day, slot_universal, m_tz, slot_local, p_name, p_group, z_link
                                ]))
                        out_strs = sorted(out_strs)
                        for out_str in out_strs:
                            f.write(','.join([
                                m_name, m_email, out_str
                                ])+'\n')
                    else:
                        f.write(','.join([
                            m_name, m_email, '', '', '', '', '', '', ''
                            ])+'\n')
                else:
                    f.write(','.join([
                        m_name, m_email, '', '', '', '', '', '', ''
                        ])+'\n')
        print('{}/{} mentors available'.format(
            available_count, self.mentor_info['mentor_num']
            ))
        print('{} mentors assigned with at least one pod, average usage {:.2%}'.format(
            assigned_count, usage_count/assigned_count
            ))


def pod_mentor_affinity(pod_info, student_abstracts, mentor_info,
                      n_components=30):
    r"""Calculates affinity matrix between pod and mentor.

    Student abstracts and mentor abstracts are embedded to a topic space. The
    pod topic is the average of student topic it contains. Euclidean distance
    between pod topic and mentor topic are thus used for affinity.

    Args
    ----
    pod_info, student_abstracts, mentor_info: dict
        Dictionaries loaded from files, check `.utils` for more details.

    Returns
    -------
    affinity: (pod_num, mentor_num), array_like
        The affinity matrix between all pods and all mentors.

    """
    s_strs = [preprocess(a) for a in student_abstracts['abstracts']]
    m_strs = [preprocess(a) for a in mentor_info['abstracts']]

    model = TfidfVectorizer(sublinear_tf=True)
    X = model.fit_transform(s_strs+m_strs)

    topic_model = PCA(n_components=n_components)
    X_topic = topic_model.fit_transform(X.todense())

    s_vecs = X_topic[:len(s_strs)]
    m_vecs = X_topic[-len(m_strs):]

    p_vecs = []
    for p_idx in range(pod_info['pod_num']):
        s_idxs = [student_abstracts['email'].index(e) for e in pod_info['student_emails'][p_idx] \
                  if e in student_abstracts['email']]
        p_vecs.append(s_vecs[s_idxs].mean(axis=0))
    p_vecs = np.array(p_vecs)

    pm_affinity = -euclidean_distances(p_vecs, m_vecs)
    return pm_affinity
