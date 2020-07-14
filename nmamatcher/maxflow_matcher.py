# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 22:45:42 2020

@author: Zhe
"""

import pandas
import numpy as np
from scipy.sparse import dok_matrix

from .utils import preprocess
from .utils import SLOT_NUM, GROUP_SLOTS, MENTOR_DSET_OPTIONS, DSET_AFFINITY

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances


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
        be found.

        """
        count = 0
        print('\nrunning Ford-Fulkerson algorithm...')
        while True:
            parent = self.get_augmenting_path(True)
            if parent[self.vertices.index('sink')]<0:
                # source is unconnected to sink on the residual flow graph
                print(f'{count} augmentation performed')
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
            print('current flow: {}'.format(self.max_flow))
            count += 1

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
    initialization. The assigned slot will occur in the third or forth gap
    between core sessions for all tracks.

    Vertices for 'pod', 'pod-slot', 'mentor-slot' and 'mentor' are created,
    with cost and capacity specified for each edge. The minimum cost maximum
    flow will be the matching result.

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
    use_second: bool
        Secondary choices from mentors are enabled. Currently, no choice
        preference is implemented, so that second choices are equivalent to
        first choices if enabled.
    affinity: (pod_num, mentor_num), array_like
        The affinity matrix between pods and mentors, e.g. based on dataset
        option compatibility.

    """
    def __init__(self, pod_info, mentor_info, max_pod_per_mentor=2,
                 use_second=False, affinity=None, special_list=None):
        self.pod_info, self.mentor_info = pod_info, mentor_info
        pod_num, mentor_num = pod_info['pod_num'], mentor_info['mentor_num']
        self.pod_num, self.mentor_num = pod_num, mentor_num
        self.max_pod_per_mentor = max_pod_per_mentor
        self.use_second = use_second
        if affinity is None:
            affinity = -np.ones((pod_num, mentor_num))
        if special_list is None:
            special_list = []

        # prepare available slots for pods
        pod_slots = []
        for p_idx, tz_group in enumerate(pod_info['tz_group']):
            slots = [] # each element is d_idx*SLOT_NUM+s_idx
            # deal with different tracks differently, so that valid slots are in the right gap
            if tz_group[-1]=='B':
                for s_idx in GROUP_SLOTS[tz_group[0]+'B+']:
                    slots.append(2*SLOT_NUM+s_idx) # second half of day 3
                for s_idx in GROUP_SLOTS[tz_group[0]+'B-']:
                    slots.append(4*SLOT_NUM+s_idx) # first half of day 5
                for s_idx in GROUP_SLOTS[tz_group]:
                    slots.append(3*SLOT_NUM+s_idx) # both halves of day 4
            elif tz_group[-1]=='A':
                for s_idx in GROUP_SLOTS[tz_group]: # day 4 and 5
                    slots.append(3*SLOT_NUM+s_idx)
                    slots.append(4*SLOT_NUM+s_idx)
            elif tz_group[-1]=='C':
                for s_idx in GROUP_SLOTS[tz_group]: # day 3 and 4
                    slots.append(2*SLOT_NUM+s_idx)
                    slots.append(3*SLOT_NUM+s_idx)
            pod_slots.append(slots)

        # prepare available slots for mentors, along with choice flexibility
        mentor_slots, mentor_flexes = [], []
        for m_idx in range(mentor_num):
            slots, flexes = [], []
            # only day 3-5 has potential match
            d_idxs = np.intersect1d(mentor_info['primary_days'][m_idx], range(2, 5))
            if mentor_info['email'][m_idx] in special_list:
                d_idxs = [special_list[mentor_info['email'][m_idx]]]
            for d_idx in d_idxs:
                for s_idx in mentor_info['primary_slots'][m_idx]:
                    slots.append(d_idx*SLOT_NUM+s_idx)
                    flexes.append(1.)
            if use_second:
                d_idxs = np.intersect1d(mentor_info['secondary_days'][m_idx], range(2, 5))
                for d_idx in d_idxs:
                    for s_idx in mentor_info['secondary_slots'][m_idx]:
                        if d_idx*SLOT_NUM+s_idx not in slots: # first choice overwrites second choice
                            slots.append(d_idx*SLOT_NUM+s_idx)
                            flexes.append(mentor_info['flexibility'][m_idx])
            mentor_slots.append(slots)
            mentor_flexes.append(flexes)

        # define vertices in the graph
        vertices = ['source', 'sink']
        for p_idx in range(pod_num):
            vertices.append(('pod', p_idx))
            for s_idx in pod_slots[p_idx]:
                vertices.append(('pod-slot', p_idx, s_idx))
        for m_idx in range(mentor_num):
            for s_idx in mentor_slots[m_idx]:
                vertices.append(('mentor-slot', m_idx, s_idx))
            vertices.append(('mentor', m_idx))

        # define edge cost and capacity in sparse matrices
        cost = dok_matrix((len(vertices), len(vertices)))
        capacity = dok_matrix((len(vertices), len(vertices)))
        # source --> pod
        u = vertices.index('source')
        for p_idx in range(pod_num):
            v = vertices.index(('pod', p_idx))
            capacity[u, v] = 1
        # pod --> pod_slot
        for p_idx in range(pod_num):
            u = vertices.index(('pod', p_idx))
            for s_idx in pod_slots[p_idx]:
                v = vertices.index(('pod-slot', p_idx, s_idx))
                capacity[u, v] = 1
        # pod-slot --> mentor-slot
        self.affinity_min, self.affinity_max = np.inf, -np.inf
        for p_idx in range(pod_num):
            for m_idx in range(mentor_num):
                for s_idx in np.intersect1d(pod_slots[p_idx], mentor_slots[m_idx]):
                    u = vertices.index(('pod-slot', p_idx, s_idx))
                    v = vertices.index(('mentor-slot', m_idx, s_idx))
                    cost[u, v] = -int(100*affinity[p_idx, m_idx]) # use int to avoid numerical negative cycle
                    cost[v, u] = -cost[u, v]
                    capacity[u, v] = 1

                    if cost[u, v]<self.affinity_min:
                        self.affinity_min = cost[u, v]
                    if cost[u, v]>self.affinity_max:
                        self.affinity_max = cost[u, v]
        # mentor-slot --> mentor
        for m_idx in range(mentor_num):
            v = vertices.index(('mentor', m_idx))
            for s_idx, flex in zip(mentor_slots[m_idx], mentor_flexes[m_idx]):
                u = vertices.index(('mentor-slot', m_idx, s_idx))
                cost[u, v] = -int(10*flex)
                cost[v, u] = -cost[u, v]
                capacity[u, v] = 1
        # mentor-->sink
        v = vertices.index('sink')
        for m_idx in range(mentor_num):
            u = vertices.index(('mentor', m_idx))
            capacity[u, v] = max_pod_per_mentor

        super(PodMentorGraph, self).__init__(vertices, cost, capacity)

    def get_matches(self, calculate_match=True):
        r"""Returns pod-mentor matches.

        Returns
        -------
        matches: list
            The matching information for each pod. Each element is a tuple like
            `(m_idx, s_idx)` containing mentor index and slot index (with day
            index encoded). If a pod is not assigned with any mentor, the list
            element is ``None``.

        """
        if calculate_match:
            self.FordFulkerson()
        flow = (self.capacity-self.residual).toarray()
        matches, count = [], 0
        for p_idx in range(self.pod_num):
            u = self.vertices.index(('pod', p_idx))
            vs, = (flow[u]>0).nonzero()
            if vs.size==1:
                u = vs[0]
                vs, = (flow[u]>0).nonzero()
                assert vs.size==1
                _, m_idx, s_idx = self.vertices[vs[0]]
                matches.append((m_idx, s_idx))
                count += 1
            elif vs.size==0:
                matches.append(None)
                print('{} not assigned with any mentor'.format(
                    self.pod_info['name'][p_idx]
                    ))
            else:
                raise RuntimeError(f'more than one outward flow detected for pod {p_idx}')
        print('\n{}/{} pods assigned with a mentor'.format(count, self.pod_num))
        return matches

    def load_mentor_schedule(self, mentor_csv, rigidity=50):
        r"""Loads from an old mentor schedule CSV file.

        Args
        ----
        mentor_csv: str
            The old CSV file, with head adjusted to current export format.

        """
        self.residual = self.capacity.copy()
        print(f'\nflow reset, loading from {mentor_csv}...')

        df = pandas.read_csv(mentor_csv)
        count, total = 0, 0
        for idx in np.random.permutation(len(df)): # randomly avoid conflicts
            if not df['email'][idx] in self.mentor_info['email']:
                continue
            else:
                m_idx = self.mentor_info['email'].index(df['email'][idx])

            if not isinstance(df['day (utc+1)'][idx], str):
                continue
            elif df['day (utc+1)'][idx]=='Wednesday':
                d_idx = 2
            elif df['day (utc+1)'][idx]=='Thursday':
                d_idx = 3
            elif df['day (utc+1)'][idx]=='Friday':
                d_idx = 4
            total += 1

            slot_str = df['slot (utc+1)'][idx]
            if slot_str.index(':')==1:
                slot_str = ' '+slot_str
            s_idx = [self._get_slot_str(i) for i in range(48)].index(slot_str)
            s_idx = d_idx*SLOT_NUM+s_idx-2

            if not df['pod'][idx] in self.pod_info['name']:
                continue
            else:
                p_idx = self.pod_info['name'].index(df['pod'][idx])

            path, is_closed = [], False
            for u in ['source', ('pod', p_idx), ('pod-slot', p_idx, s_idx),
                      ('mentor-slot', m_idx, s_idx), ('mentor', m_idx), 'sink']:
                if u in self.vertices:
                    path.append(self.vertices.index(u))
                else:
                    is_closed = True
            if is_closed:
                continue

            for u, v in zip(path[:-1], path[1:]):
                if self.residual[u, v]<=0:
                    is_closed = True
            if is_closed:
                continue

            for u, v in zip(path[:-1], path[1:]):
                self.residual[u, v] -= 1
                self.residual[v, u] += 1

                self.cost[u, v] = self.affinity_min-rigidity
                self.cost[v, u] = -self.cost[u, v]
            count += 1
        print(f'{count}/{total} matches loaded')

    def _get_day_str(self, d_idx):
        if d_idx==2:
            day_str = 'Wednesday'
        elif d_idx==3:
            day_str = 'Thursday'
        elif d_idx==4:
            day_str = 'Friday'
        return day_str

    def _get_slot_str(self, s_idx):
        assert s_idx>=-48 and s_idx<96
        if s_idx<0:
            slot_str = ' (-1)'
            s_idx += 48
        elif s_idx>=48:
            slot_str = ' (+1)'
            s_idx -= 48
        else:
            slot_str = ''
        def hour_label(h_idx):
            h_label = ' AM' if h_idx<24 or h_idx==48 else ' PM'
            h_idx %= 24
            h_label = '{:2d}:{:02d}'.format(
                ((h_idx//2)-1)%12+1, (h_idx%2)*30,
                )+h_label
            return h_label
        slot_str = '{} - {}'.format(
            hour_label(s_idx), hour_label(s_idx+1)
            )+slot_str
        return slot_str

    def export_schedules(self, r_id, calculate_match=True):
        r"""Exports pod schedule and mentor schedule CSV files.

        Args
        ----
        r_id: str
            The random ID for identifying output files.

        """
        matches = self.get_matches(calculate_match)

        # export pod schedule
        with open(f'pod.schedule_{r_id}.csv', 'w') as f:
            f.write('pod,pod time zone group,day (utc+1),slot (utc+1),mentor,mentor e-mail,zoom link\n')
            for p_idx in range(self.pod_num):
                if matches[p_idx]:
                    m_idx, s_idx = matches[p_idx]
                    # convert from UTC to UTC+1
                    s_idx += 2
                    d_idx = s_idx//SLOT_NUM
                    s_idx = s_idx%SLOT_NUM
                    f.write('{},{},{},{},{},{},\n'.format(
                        self.pod_info['name'][p_idx],
                        self.pod_info['tz_group'][p_idx],
                        self._get_day_str(d_idx), self._get_slot_str(s_idx),
                        ' '.join([
                        self.mentor_info['first_name'][m_idx],
                        self.mentor_info['last_name'][m_idx]
                        ]),
                        self.mentor_info['email'][m_idx]
                        ))
                else:
                    f.write('{},{},{},{},{},{},\n'.format(
                        self.pod_info['name'][p_idx],
                        self.pod_info['tz_group'][p_idx],
                        'N/A', 'N/A', 'N/A', 'N/A'
                        ))

        available_count, assigned_count, usage_count = 0, 0, 0
        with open(f'mentor.schedule_{r_id}.csv', 'w') as f:
            f.write('name,email,day (utc+1),slot (utc+1),mentor time zone,slot (local),pod,pod time zone group,zoom link\n')
            for m_idx in range(self.mentor_num):
                m_name = ' '.join([
                    self.mentor_info['first_name'][m_idx],
                    self.mentor_info['last_name'][m_idx]
                    ])
                m_email = self.mentor_info['email'][m_idx]

                m_capacity = self.capacity[:, self.vertices.index(('mentor', m_idx))].toarray().sum()
                if m_capacity>0:
                    available_count += 1

                    assignments = []
                    for p_idx, match in enumerate(matches): # (m_idx, s_idx)
                        if match and match[0]==m_idx:
                            assignments.append((p_idx, match[1]))

                    if assignments:
                        assigned_count += 1
                        usage_count += len(assignments)/m_capacity

                        out_strs = []
                        for p_idx, s_idx in assignments:
                            # convert from UTC to UTC+1
                            s_idx += 2
                            d_idx = s_idx//SLOT_NUM
                            s_idx = s_idx%SLOT_NUM

                            day_str = self._get_day_str(d_idx)
                            slot_str_universal = self._get_slot_str(s_idx)
                            if slot_str_universal.startswith(' '):
                                slot_str_universal = slot_str_universal[1:]

                            # convert to local time zone
                            s_idx -= 2
                            m_tz = self.mentor_info['timezone'][m_idx]
                            if m_tz[3]=='+':
                                s_idx += int(m_tz[4:])*2
                            else:
                                s_idx -= int(m_tz[4:])*2
                            slot_str_local = self._get_slot_str(s_idx)
                            if slot_str_local.startswith(' '):
                                slot_str_local = slot_str_local[1:]

                            p_name = self.pod_info['name'][p_idx]
                            p_group = self.pod_info['tz_group'][p_idx]

                            z_link = ''

                            out_strs.append(','.join([
                                day_str, slot_str_universal, m_tz,
                                slot_str_local, p_name, p_group, z_link
                                ]))
                        for out_str in sorted(out_strs):
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
        print('{}/{} mentors available for week 1'.format(
            available_count, self.mentor_num
            ))
        print('{} mentors assigned with at least one pod, average usage {:.2%}'.format(
            assigned_count, usage_count/assigned_count
            ))


def pod_mentor_topic_affinity(pod_info, mentor_info, student_abstracts,
                              n_components=30):
    r"""Calculates affinity matrix between pod and mentor based on abstract
    topic.

    Student abstracts and mentor abstracts are embedded to a topic space. The
    pod topic is the average of student topic it contains. Euclidean distance
    between pod topic and mentor topic are thus used for affinity.

    Args
    ----
    pod_info, mentor_info, student_abstracts: dict
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

    pm_affinity = -cosine_distances(p_vecs, m_vecs)
    return pm_affinity


def pod_mentor_dset_affinity(pod_info, mentor_info, base_score=1.):
    r"""Calculates affinity matrix between pod and mentor based on dataset
    option.

    For each pod and mentor, a score is calculated based on the affinity of
    pod dataset option and the preferred dataset options of the mentor.

    Args
    ----
    pod_info, mentor_info: dict
        Dictionaries loaded from files, check `.utils` for more details.
    base_score: float
        Base score for each pod-mentor pair. Mentors provide dataset options
        can add corresponding affinity scores onto the base score.

    Returns
    -------
    affinity: (pod_num, mentor_num), array_like
        The affinity matrix between all pods and all mentors.

    """
    pod_num, mentor_num = pod_info['pod_num'], mentor_info['mentor_num']
    pm_affinity = np.ones((pod_num, mentor_num))*base_score

    for p_idx in range(pod_num):
        dset_affinity = DSET_AFFINITY[pod_info['dset_option'][p_idx]]
        for m_idx in range(mentor_num):
            for dset_opt in mentor_info['dset_option'][m_idx]:
                pm_affinity[p_idx, m_idx] += dset_affinity[MENTOR_DSET_OPTIONS.index(dset_opt)]
    return pm_affinity
