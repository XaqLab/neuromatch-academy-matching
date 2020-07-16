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
    def __init__(self, vertices, cost, capacity, is_aux=False):
        assert set(['source', 'sink']).issubset(set(vertices))
        if is_aux:
            assert set(['_source', '_sink']).issubset(set(vertices))
        self.is_aux = is_aux
        N = len(set(vertices))
        assert len(vertices)==N, 'duplicate vertices detected'
        assert cost.shape==(N, N) and capacity.shape==(N, N)

        self.vertices = vertices
        self.cost, self.capacity = cost, capacity
        self.residual = capacity.copy() # the residual flow graph
        self.max_flow = 0.

    def get_augmenting_path(self, s, random_order=False):
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
        d: (N,), array_like
            The minimum distance of each vertices to source.
        parent: (N,), array_like
            The parent of each vertices along the shortest path to the source.

        """
        d = np.empty((len(self.vertices),), np.float) # distance to source
        d.fill(np.inf)
        parent = np.empty((len(self.vertices),), np.int) # parent index along the shortest path
        parent.fill(-1)

        queue, d[s] = [s], 0
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
        return d, parent

    def FordFulkerson(self, show_progress=True):
        r"""Implements Ford-Fulkerson algorithm to find min-cost max-flow.

        Augmenting paths are repeatedly calculated and used to update the
        residual flow graph. Algorithm terminates when no augmenting path can
        be found.

        """
        if self.is_aux:
            s = self.vertices.index('_source')
            t = self.vertices.index('_sink')
        else:
            s = self.vertices.index('source')
            t = self.vertices.index('sink')
        count = 0
        print('\nrunning Ford-Fulkerson algorithm...')
        while True:
            d, parent = self.get_augmenting_path(s, True)
            if parent[t]<0:
                # source is unconnected to sink on the residual flow graph
                print(f'{count} augmentation performed')
                break
            if show_progress:
                self.print_path(d, parent, s, t)

            # calculate flow increase
            delta_flow = np.inf
            v = t
            while v!=s:
                u = parent[v]
                delta_flow = min(delta_flow, self.residual[u, v])
                v = u
            self.max_flow += delta_flow
            if show_progress:
                print('current flow: {}'.format(self.max_flow))
            count += 1

            # update residual
            v = t
            while v!=s:
                u = parent[v]
                self.residual[u, v] -= delta_flow # forward arc
                self.residual[v, u] += delta_flow # backward arc
                v = u

    def print_path(self, d, parent, s, t):
        r"""Prints a path from source to sink.

        Args
        ----
        d: (N,), array_like
            The minimum distance of each vertices to source.
        parent: (N,), array_like
            The parent of each vertices along the shortest path to the source.

        """
        idxs = [t]
        while idxs[-1]!=s:
            idxs.append(parent[idxs[-1]])
        idxs.reverse()
        print('[{:g}]'.format(d[t])+' '+'-->'.join([str(self.vertices[i]) for i in idxs]))


class PodMentorGraph(FlowGraph):
    r"""Flow graph for pod-mentor matching.

    During week 1 of NMA, each pod is assigned with a mentor for project
    initialization. The assigned slot will occur in the third or forth gap
    between core sessions for all tracks.

    Vertices for 'pod', 'mentor-slot' and 'mentor' are created, with cost and
    capacity specified for each edge. The minimum cost maximum flow will be the
    matching result.

    Args
    ----
    pod_info: dict
        The dictionary containing information of all pods returned by
        `.utils.load_pod_info`.
    mentor_info: dict
        The dictionary containing available time of all mentors returned by
        `.utils.load_mentor_info`.
    min_mentor_per_pod: int
        The minimum number of mentors assigned to a pod.
    max_mentor_per_pod: int
        The maximum number of mentors assigned to a pod.
    min_pod_per_mentor: int
        The minimum number of pods assigned to a mentor.
    max_pod_per_mentor: int
        The maximum number of pods assigned to a mentor.
    use_second: bool
        Secondary choices from mentors are enabled. Currently, no choice
        preference is implemented, so that second choices are equivalent to
        first choices if enabled.
    affinity: (pod_num, mentor_num), array_like
        The affinity matrix between pods and mentors, e.g. based on dataset
        option compatibility.
    fixed_matches: list
        The list of matches that must be included, in the form of
        `(s_idx, pod_name, mentor_email)`. Lower limit for the corresponding
        edge is set to 1.
    mentor_requests: dict
        The dictionary containing mentor requests returned by
        `.utils.load_mentor_requests`.
    now_idx: int
        The slot index used to determine past and future, in UTC. Slots in the
        past will not be activated.

    """
    def __init__(self, pod_info, mentor_info,
                 min_mentor_per_pod=0, max_mentor_per_pod=1,
                 min_pod_per_mentor=0, max_pod_per_mentor=2,
                 use_second=False, affinity=None, fixed_matches=None,
                 mentor_requests=None, now_idx=0):
        self.pod_info, self.mentor_info = pod_info, mentor_info
        pod_num, mentor_num = pod_info['pod_num'], mentor_info['mentor_num']
        self.pod_num, self.mentor_num = pod_num, mentor_num
        self.min_mpp = min_mentor_per_pod
        self.max_mpp = max_mentor_per_pod
        self.min_ppm = min_pod_per_mentor
        self.max_ppm = max_pod_per_mentor
        self.use_second = use_second
        if affinity is None:
            affinity = np.ones((pod_num, mentor_num))
        if fixed_matches is None:
            fixed_matches = []
        self.fixed_matches = fixed_matches
        if mentor_requests is None:
            mentor_requests = {
                'request_num': 0,
                'email': [], 'type': [], 'd_idx': [], 's_idx': [],
                }

        # preprocess requests
        to_deact = []
        to_add, to_remove = {}, {}
        for r_idx in range(mentor_requests['request_num']):
            m_idx = self.mentor_info['email'].index(mentor_requests['email'][r_idx])
            if mentor_requests['type'][r_idx]=='deactivate':
                to_deact.append(m_idx)
            else:
                s_idx = int(mentor_requests['d_idx'][r_idx]*SLOT_NUM+mentor_requests['s_idx'][r_idx])
                if mentor_requests['type'][r_idx]=='add':
                    if m_idx in to_add:
                        to_add[m_idx].append(s_idx)
                    else:
                        to_add[m_idx] = [s_idx]
                if mentor_requests['type'][r_idx]=='remove':
                    if m_idx in to_remove:
                        to_remove[m_idx].append(s_idx)
                    else:
                        to_remove[m_idx] = [s_idx]

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
            slots = [s for s in slots if s>=now_idx]
            pod_slots.append(slots)

        # prepare available slots for mentors, along with choice flexibility
        mentor_slots, mentor_flexes = [], []
        for m_idx in range(mentor_num):
            slots, flexes = [], []
            if m_idx not in to_deact:
                d_idxs = np.intersect1d(mentor_info['primary_days'][m_idx], range(2, 5)) # only day 3-5 has potential match
                # add first choices
                for d_idx in d_idxs:
                    for s_idx in mentor_info['primary_slots'][m_idx]:
                        slots.append(d_idx*SLOT_NUM+s_idx)
                        flexes.append(1.)
                # add second choices
                if use_second:
                    d_idxs = np.intersect1d(mentor_info['secondary_days'][m_idx], range(2, 5))
                    for d_idx in d_idxs:
                        for s_idx in mentor_info['secondary_slots'][m_idx]:
                            if d_idx*SLOT_NUM+s_idx not in slots: # first choice overwrites second choice
                                slots.append(d_idx*SLOT_NUM+s_idx)
                                flexes.append(mentor_info['flexibility'][m_idx])
                # deal with requests
                if m_idx in to_add:
                    for s in to_add[m_idx]:
                        if s not in slots:
                            slots.append(s)
                            flexes.append(1.)
                if m_idx in to_remove:
                    sf_pairs = [(s, f) for s, f in zip(slots, flexes) if s in to_remove[m_idx]]
                    if sf_pairs:
                        slots, flexes = map(list, zip(*sf_pairs))
                    else:
                        slots, flexes = [], []
            sf_pairs = [(s, f) for s, f in zip(slots, flexes) if s>=now_idx]
            if sf_pairs:
                slots, flexes = map(list, zip(*sf_pairs))
            else:
                slots, flexes = [], []
            mentor_slots.append(slots)
            mentor_flexes.append(flexes)

        # update pod_slots, mentor_slots from fixed match
        for s_idx, pod_name, mentor_email in fixed_matches:
            if pod_name in pod_info['name']:
                p_idx = pod_info['name'].index(pod_name)
            else:
                p_idx = None
                print(f'{pod_name} not recognized')
            if mentor_email in mentor_info['email']:
                m_idx = mentor_info['email'].index(mentor_email)
            else:
                m_idx = None
                print(f'{mentor_email} not recognized')
            if p_idx is None or m_idx is None:
                print('({}, {}, {}) will be ignored'.format(
                    s_idx, pod_name, mentor_email
                    ))
                continue
            if s_idx not in pod_slots[p_idx]:
                pod_slots[p_idx].append(s_idx)
            if s_idx not in mentor_slots[m_idx]:
                mentor_slots[m_idx].append(s_idx)
                mentor_flexes[m_idx].append(1.)

        # define vertices in the graph
        vertices = ['source', 'sink']
        for p_idx in range(pod_num):
            vertices.append(('pod', p_idx))
        for m_idx in range(mentor_num):
            for s_idx in mentor_slots[m_idx]:
                vertices.append(('mentor-slot', m_idx, s_idx))
            vertices.append(('mentor', m_idx))

        # define edge cost and capacity in sparse matrices
        cost = dok_matrix((len(vertices), len(vertices)))
        lower_bound = dok_matrix((len(vertices), len(vertices)))
        upper_bound = dok_matrix((len(vertices), len(vertices)))
        # source --> pod
        u = vertices.index('source')
        for p_idx in range(pod_num):
            v = vertices.index(('pod', p_idx))
            lower_bound[u, v] = min_mentor_per_pod
            upper_bound[u, v] = max_mentor_per_pod
        # pod --> mentor-slot
        self.shared_slots = np.zeros((pod_num, mentor_num), np.object)
        for p_idx in range(pod_num):
            for m_idx in range(mentor_num):
                self.shared_slots[p_idx, m_idx] = []
                for s_idx in np.intersect1d(pod_slots[p_idx], mentor_slots[m_idx]):
                    u = vertices.index(('pod', p_idx))
                    v = vertices.index(('mentor-slot', m_idx, s_idx))
                    cost[u, v] = int(100*(2-affinity[p_idx, m_idx])) # use int to avoid numerical negative cycle
                    cost[v, u] = -cost[u, v]
                    upper_bound[u, v] = 1

                    self.shared_slots[p_idx, m_idx].append((s_idx, u, v))
        # mentor-slot --> mentor
        for m_idx in range(mentor_num):
            v = vertices.index(('mentor', m_idx))
            for s_idx, flex in zip(mentor_slots[m_idx], mentor_flexes[m_idx]):
                u = vertices.index(('mentor-slot', m_idx, s_idx))
                cost[u, v] = int(10*(1-flex))
                cost[v, u] = -cost[u, v]
                upper_bound[u, v] = 1
        # mentor-->sink
        v = vertices.index('sink')
        for m_idx in range(mentor_num):
            u = vertices.index(('mentor', m_idx))
            lower_bound[u, v] = min_pod_per_mentor
            upper_bound[u, v] = max_pod_per_mentor

        # add fixed matches
        for s_idx, pod_name, mentor_email in fixed_matches:
            if not (pod_name in pod_info['name'] and mentor_email in mentor_info['email']):
                continue
            p_idx = pod_info['name'].index(pod_name)
            m_idx = mentor_info['email'].index(mentor_email)
            u = vertices.index(('pod', p_idx))
            v = vertices.index(('mentor-slot', m_idx, s_idx))
            lower_bound[u, v] = 1

        self.lower_bound, self.upper_bound = lower_bound, upper_bound
        super(PodMentorGraph, self).__init__(vertices, cost, upper_bound-lower_bound)
        print('\npod-mentor graph initialized')
        print('{} pods, {} mentors, max {} mentor(s) per pod, max {} pod(s) per mentor'.format(
            pod_num, mentor_num, max_mentor_per_pod, max_pod_per_mentor
            ))

    @staticmethod
    def _get_day_str(d_idx, abb=False):
        if abb:
            if d_idx==2:
                day_str = 'WED'
            elif d_idx==3:
                day_str = 'THU'
            elif d_idx==4:
                day_str = 'FRI'
        else:
            if d_idx==2:
                day_str = 'Wednesday'
            elif d_idx==3:
                day_str = 'Thursday'
            elif d_idx==4:
                day_str = 'Friday'
        return day_str

    @staticmethod
    def _get_slot_str(s_idx):
        assert s_idx>=-SLOT_NUM and s_idx<2*SLOT_NUM
        if s_idx<0:
            slot_str = ' (-1)'
            s_idx += SLOT_NUM
        elif s_idx>=SLOT_NUM:
            slot_str = ' (+1)'
            s_idx -= SLOT_NUM
        else:
            slot_str = ''
        def hour_label(h_idx):
            h_label = ' AM' if h_idx<SLOT_NUM/2 or h_idx==SLOT_NUM else ' PM'
            h_idx %= SLOT_NUM/2
            h_label = '{:d}:{:02d}'.format(
                int(((h_idx//2)-1)%12+1), int((h_idx%2)*30),
                )+h_label
            return h_label
        slot_str = '{} - {}'.format(
            hour_label(s_idx), hour_label(s_idx+1)
            )+slot_str
        return slot_str

    def update_flow(self):
        r"""Recalculates the flow based current cost, lower/upper bounds.

        To account for lower bounds on the edges, an auxiliary graph is first
        constructed, by adding ``'_source'`` and ``'_sink'`` to the original
        graph. Capacity from ``'_source'`` to all original vertices will be
        the sum of lower bounds of inward edges. Capacity from all original
        vertices to ``'_sink'`` will be the sum of upper bounds of outward
        edges. The max flow on this auxillary flow will be used as an
        initialization for improving on original graph.

        """
        N = len(self.vertices)
        _vertices = self.vertices+['_source', '_sink']
        s, t = _vertices.index('_source'), _vertices.index('_sink')
        cost, capacity = dok_matrix((N+2, N+2)), dok_matrix((N+2, N+2))

        cost[:N, :N] = self.cost
        capacity[:N, :N] = self.upper_bound-self.lower_bound
        # _source to main vertices
        l_in = self.lower_bound.toarray().sum(axis=0)
        us, = l_in.nonzero()
        for u in us:
            capacity[s, u] = l_in[u]
        # main vertices to _sink
        l_out = self.lower_bound.toarray().sum(axis=1)
        us, = l_out.nonzero()
        for u in us:
            capacity[u, t] = l_out[u]
        # sink to source
        infinite_flow = self.upper_bound.toarray().sum()
        capacity[_vertices.index('sink'), _vertices.index('source')] = infinite_flow

        # get a feasible flow on original graph by finding the max flow on
        # auxiliary graph
        aux_fg = FlowGraph(_vertices, cost, capacity, True)
        aux_fg.FordFulkerson()
        self.residual = aux_fg.residual[:N, :N]
        s, t = self.vertices.index('source'), self.vertices.index('sink')
        self.residual[s, t] = 0
        self.residual[t, s] = 0

        self.FordFulkerson()

    def get_matches(self, update_flow=True):
        r"""Returns pod-mentor matches.

        Args
        ----
        update_flow: bool
            Run Ford-Fulkerson algorithm to find maximum flow when ``True``.

        Returns
        -------
        matches: list
            The matched slots for pod-mentor session in week 1. Each element is
            a tuple like `(s_idx, pod_name, mentor_email)` containing slot
            index (with day index encoded), pod name and mentor e-mail address.

        """
        if update_flow:
            self.update_flow()
        flow = (self.capacity-self.residual).toarray()

        matches = self.fixed_matches.copy()
        for p_idx in range(self.pod_num):
            for m_idx in range(self.mentor_num):
                for s_idx, u, v in self.shared_slots[p_idx, m_idx]:
                    if flow[u, v]>0:
                        matches.append(
                            (s_idx, self.pod_info['name'][p_idx], self.mentor_info['email'][m_idx])
                            )
        return matches

    def get_pod_centered_view(self, matches, print_hanging=True):
        r"""Returns pod-centered view of matches.

        Args
        ----
        matches: list
            The matched slots for pod-mentor session in week 1. Each element is
            a tuple like `(s_idx, pod_name, mentor_email)`.
        print_hanging: bool
            Print unassigned pods when ``True``.

        Returns
        -------
        p_matches: dict
            A dict of matched mentors for all pods with pod name as key, and
            the slot and mentor index `(s_idx, m_idx)` as value.

        """
        matched = np.zeros((self.pod_num,), np.float)
        p_matches = dict((pod_name, []) for pod_name in self.pod_info['name'])
        for s_idx, pod_name, mentor_email in matches:
            if pod_name in self.pod_info['name']:
                p_idx = self.pod_info['name'].index(pod_name)
                matched[p_idx] += 1
            else:
                p_idx = None
            if mentor_email in self.mentor_info['email']:
                m_idx = self.mentor_info['email'].index(mentor_email)
            else:
                m_idx = None

            if pod_name not in p_matches: # probably removed pods
                p_matches[pod_name] = []
            p_matches[pod_name].append((s_idx, m_idx))

        print('{}/{} pods assigned with a mentor'.format((matched>0).sum(), self.pod_num))
        print('{:.2f} mentors assigned to each pod on average'.format(matched.mean()))
        if print_hanging and np.any(matched==0):
            print('hanging pods:')
            for p_idx in range(self.pod_num):
                if matched[p_idx]==0:
                    print(self.pod_info['name'][p_idx])
        return p_matches

    def get_mentor_centered_view(self, matches, print_idle=False):
        r"""Returns mentor-centered view of matches.

        Args
        ----
        matches: list
            The matched slots for pod-mentor session in week 1. Each element is
            a tuple like `(s_idx, pod_name, mentor_email)`.
        print_idle: bool
            Print unassigned but available mentors when ``True``.

        Returns
        -------
        m_matches: dict
            A dict of matched pods for all mentors with e-mail address as key,
            and the slot and pod index `(s_idx, p_idx)` as value.

        """
        usage = np.zeros((self.mentor_num,), np.float)
        limit = np.zeros((self.mentor_num,), np.float)
        for m_idx in range(self.mentor_num):
            u = self.vertices.index(('mentor', m_idx))
            limit[m_idx] = self.capacity[:, u].toarray().sum()

        m_matches = dict((mentor_email, []) for mentor_email in self.mentor_info['email'])
        for s_idx, pod_name, mentor_email in matches:
            if pod_name in self.pod_info['name']:
                p_idx = self.pod_info['name'].index(pod_name)
            else:
                p_idx = None
            if mentor_email in self.mentor_info['email']:
                m_idx = self.mentor_info['email'].index(mentor_email)
                usage[m_idx] += 1
            else:
                m_idx = None

            if mentor_email not in m_matches: # probably removed mentors
                m_matches[mentor_email] = []
            m_matches[mentor_email].append((s_idx, p_idx))

        print('{}/{} mentors assigned to at least a pod'.format((usage>0).sum(), self.mentor_num))
        print('{:.2%} usage for the busy mentors'.format((usage/(limit+1e-8))[usage>0].mean()))
        if print_idle and np.any(np.logical_and(usage==0, limit>0)):
            print('idle mentors:')
            count = 0
            for m_idx in range(self.mentor_num):
                if usage[m_idx]==0 and limit[m_idx]>0:
                    count += 1
                    print(' '.join([
                        self.mentor_info['first_name'][m_idx],
                        self.mentor_info['last_name'][m_idx]+',',
                        '('+self.mentor_info['email'][m_idx]+')'
                        ]))
            print('\n{} mentors available but not assigned'.format(count))
        return m_matches

    def export_pod_schedule(self, r_id, matches=None,
                            update_flow=True, print_hanging=True):
        r"""Exports pod schedule CSV file.

        Args
        ----
        r_id: str
            The random ID for identifying output files.
        matches: list
            A list of tuples `(s_idx, pod_name, mentor_email)` if provided, see
            `get_matches` for more details. If `matches` is ``None``, matches
            will be calculated based on the current flow.
        update_flow: bool
            Run Ford-Fulkerson algorithm to find maximum flow when ``True``.
            Used only when `matches` is ``None``.
        print_hanging: bool
            Print unassigned pod when ``True``.

        """
        if matches is None:
            matches = self.get_matches(update_flow)

        p_matches = self.get_pod_centered_view(matches, print_hanging)
        out_strs = []
        for pod_name in sorted(self.pod_info['name']):
            p_idx = self.pod_info['name'].index(pod_name)
            if p_matches[pod_name]:
                _out_strs = []
                for s_idx, m_idx in p_matches[pod_name]:
                    s_idx += 2 # convert from UTC to UTC+1
                    d_idx = s_idx//SLOT_NUM
                    s_idx = s_idx%SLOT_NUM

                    _out_strs.append('{},{},{},{},{},{},\n'.format(
                        pod_name, self.pod_info['tz_group'][p_idx],
                        self._get_day_str(d_idx), self._get_slot_str(s_idx),
                        ' '.join([
                            self.mentor_info['first_name'][m_idx],
                            self.mentor_info['last_name'][m_idx]
                            ]),
                        self.mentor_info['email'][m_idx],
                        ))
                out_strs += sorted(_out_strs)
            else:
                out_strs.append('{},{},{},{},{},{},\n'.format(
                    pod_name, self.pod_info['tz_group'][p_idx],
                    '', '', '', '',
                    ))

        with open(f'pod.schedule_{r_id}.csv', 'w') as f:
            f.write('pod,pod time zone group,day (utc+1),slot (utc+1),mentor,mentor e-mail,zoom link\n')
            for out_str in out_strs:
                f.write(out_str)

    def export_mentor_schedule(self, r_id, matches=None,
                               update_flow=True, print_idle=True):
        r"""Exports mentor schedule CSV file.

        Args
        ----
        r_id: str
            The random ID for identifying output files.
        matches: list
            A list of tuples `(s_idx, pod_name, mentor_email)` if provided, see
            `get_matches` for more details. If `matches` is ``None``, matches
            will be calculated based on the current flow.
        update_flow: bool
            Run Ford-Fulkerson algorithm to find maximum flow when ``True``.
            Used only when `matches` is ``None``.
        print_idle: bool
            Print unassigned mentor when ``True``.

        """
        if matches is None:
            matches = self.get_matches(update_flow)

        m_matches = self.get_mentor_centered_view(matches, print_idle)
        out_strs = []
        for mentor_email in sorted(self.mentor_info['email']):
            m_idx = self.mentor_info['email'].index(mentor_email)
            if m_matches[mentor_email]:
                _out_strs = []
                for s_idx, p_idx in m_matches[mentor_email]:
                    s_idx += 2 # convert from UTC to UTC+1
                    d_idx = s_idx//SLOT_NUM
                    s_idx = s_idx%SLOT_NUM

                    day_str = self._get_day_str(d_idx)
                    slot_str_universal = self._get_slot_str(s_idx)

                    # convert to local time zone, keep day the same
                    s_idx -= 2
                    m_tz = self.mentor_info['timezone'][m_idx]
                    if m_tz[3]=='+':
                        s_idx += int(m_tz[4:])*2
                    else:
                        s_idx -= int(m_tz[4:])*2
                    slot_str_local = self._get_slot_str(s_idx)

                    _out_strs.append(
                        '{},{},{},{},{},{},{},{},\n'.format(
                            ' '.join([
                                self.mentor_info['first_name'][m_idx],
                                self.mentor_info['last_name'][m_idx]
                                ]),
                            mentor_email, day_str, slot_str_universal,
                            m_tz, slot_str_local, self.pod_info['name'][p_idx],
                            self.pod_info['tz_group'][p_idx],
                            )
                        )
                out_strs += sorted(_out_strs)
            else:
                out_strs.append('{},{},{},{},{},{},{},{},\n'.format(
                    ' '.join([
                        self.mentor_info['first_name'][m_idx],
                        self.mentor_info['last_name'][m_idx]
                        ]),
                    mentor_email, '', '', '', '', '', '',
                    ))

        with open(f'mentor.schedule_{r_id}.csv', 'w') as f:
            f.write('name,email,day (utc+1),slot (utc+1),mentor time zone,slot (local),pod,pod time zone group,zoom link\n')
            for out_str in out_strs:
                f.write(out_str)

    @classmethod
    def read_schedule(cls, r_id, csv_type='mentor', now_idx=0):
        r"""Reads matches from a schedule CSV file.

        Args
        ----
        r_id: str
            The random ID for identifying output files.
        now_idx: int
            The slot index used to determine past and future, in UTC.

        Returns
        -------
        matches_past, matches_future: list
            Each is a list of tuples as `(s_idx, pod_name, mentor_email)`.

        """
        assert csv_type in ['pod', 'mentor']
        df = pandas.read_csv(f'{csv_type}.schedule_{r_id}.csv')
        day_strs = [cls._get_day_str(d_idx) for d_idx in range(2, 5)]
        slot_strs = [cls._get_slot_str(s_idx) for s_idx in range(SLOT_NUM)]
        matches_past, matches_future = [], []
        for i in range(len(df)):
            if not isinstance(df['day (utc+1)'][i], str):
                continue
            day_str = df['day (utc+1)'][i]
            slot_str = df['slot (utc+1)'][i]
            pod_name = df['pod'][i]
            if csv_type=='pod':
                mentor_email = df['mentor e-mail'][i]
            if csv_type=='mentor':
                mentor_email = df['email'][i]

            d_idx = day_strs.index(day_str)+2
            s_idx = slot_strs.index(slot_str)
            s_idx = d_idx*SLOT_NUM+s_idx-2 # convert from UCT+1 to UTC
            if s_idx<now_idx:
                matches_past.append((s_idx, pod_name, mentor_email))
            else:
                matches_future.append((s_idx, pod_name, mentor_email))
        return matches_past, matches_future

    def export_changelog(self, c_id, matches_old, matches_new):
        r"""Exports change log between two matches.

        A change log is generated for mentors.

        Args
        ----
        c_id: str
            The ID for identifying change log files. Usually in the form
            ``[old_ID]-[new_ID]``.
        matches_old, matches_new: list
            The old and new matches, both are the list containing
            `(s_idx, pod_name, mentor_email)` tuples.

        """
        def entry_str(s_idx, p_idx):
            s_idx += 2 # convert from UTC to UTC+1
            d_idx = s_idx//SLOT_NUM
            s_idx = s_idx%SLOT_NUM
            return '{} {}, {}\n'.format(
                self._get_day_str(d_idx), self._get_slot_str(s_idx),
                self.pod_info['name'][p_idx],
                )

        print('\npreview of old and new matches')
        print('[old]:')
        m_matches_old = self.get_mentor_centered_view(matches_old)
        print('[new]:')
        m_matches_new = self.get_mentor_centered_view(matches_new)
        with open(f'change.log_{c_id}.txt', 'w') as f:
            f.write('Change Log\n')
            for mentor_email in self.mentor_info['email']:
                set_old = set(m_matches_old[mentor_email]) if mentor_email in m_matches_old else set()
                set_new = set(m_matches_new[mentor_email]) if mentor_email in m_matches_new else set()

                if set_new!=set_old:
                    m_idx = self.mentor_info['email'].index(mentor_email)
                    f.write('\n{} {}, {}, #pod {} --> {}\n'.format(
                        self.mentor_info['first_name'][m_idx],
                        self.mentor_info['last_name'][m_idx],
                        mentor_email, len(set_old), len(set_new)
                        ))
                    to_remove = set_old.difference(set_new)
                    to_add = set_new.difference(set_old)
                    if to_remove:
                        f.write('remove\n')
                        for s_idx, p_idx in to_remove:
                            f.write(entry_str(s_idx, p_idx))
                    if to_add:
                        f.write('add\n')
                        for s_idx, p_idx in to_add:
                            f.write(entry_str(s_idx, p_idx))

    def mark_matches(self, matches, cost_change=100):
        r"""Loads existing matches to the flow graph.

        After resetting the flow, matches are randomly selected and attempted
        to add on to the flow graph. If a match is feasible, the cost along
        path of this match is decreased, so that later flow update will favor
        this path.

        Args
        ----
        matches: list
            A list of tuples as `(s_idx, pod_name, mentor_email)`.
        volatility: float
            A number controls how volatile the loaded match is. Higher value
            means more difficulty to shift the flow.

        """
        for s_idx, pod_name, mentor_email in matches:
            if pod_name in self.pod_info['name']:
                p_idx = self.pod_info['name'].index(pod_name)
            else:
                print(f'{pod_name} not found in pod names')
                continue
            if mentor_email in self.mentor_info['email']:
                m_idx = self.mentor_info['email'].index(mentor_email)
            else:
                print(f'{mentor_email} not found in mentor e-mail addresses')
                continue

            u = self.vertices.index(('pod', p_idx))
            v = self.vertices.index(('mentor-slot', m_idx, s_idx))
            self.cost[u, v] -= cost_change
            self.cost[v, u] = -self.cost[u, v]


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
