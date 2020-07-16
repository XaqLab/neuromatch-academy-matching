# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

import argparse, time

from nmamatcher.utils import load_pod_info, load_mentor_info, load_mentor_requests, load_student_abstracts
from nmamatcher.utils import random_id
from nmamatcher.maxflow_matcher import PodMentorGraph
from nmamatcher.maxflow_matcher import pod_mentor_topic_affinity, pod_mentor_dset_affinity

parser = argparse.ArgumentParser()
parser.add_argument('--affinity_type', default='dataset', choices=['abstract', 'dataset'])
args = parser.parse_args()

if __name__=='__main__':
    print('loading pod information...')
    pod_info = load_pod_info('pod.map.csv')
    print('loading mentor information...')
    mentor_info = load_mentor_info('mentors.info.xlsx')
    print('loading mentor requests...')
    mentor_requests = load_mentor_requests('mentor.requests.xlsx')

    if args.affinity_type=='abstract':
        student_abstracts = load_student_abstracts('student.abstracts.csv')
        affinity = pod_mentor_topic_affinity(pod_info, mentor_info, student_abstracts) # stochastic
    if args.affinity_type=='dataset':
        affinity = pod_mentor_dset_affinity(pod_info, mentor_info)

    now_idx = 138 # day 3 slot 42, Wed 21:00 UTC
    old_id = 'C56E'
    old_matches_past, old_matches_future = PodMentorGraph.read_schedule(old_id, now_idx=now_idx)

    print('initializing graph model...')
    pmg = PodMentorGraph(pod_info, mentor_info, mentor_requests=mentor_requests, fixed_matches=old_matches_past,
                         min_mentor_per_pod=1, max_mentor_per_pod=3,
                         min_pod_per_mentor=0, max_pod_per_mentor=3)
    pmg.mark_matches(old_matches_future)

    matches = pmg.get_matches()
    r_id = random_id()
    pmg.export_pod_schedule(r_id, matches)
    pmg.export_mentor_schedule(r_id, matches)

    # # gather 2-pod mentors
    # pmg = PodMentorGraph(pod_info, mentor_info, max_mentor_per_pod=1, max_pod_per_mentor=3,
    #                      mentor_requests=mentor_requests, affinity=None)
    # old_matches = PodMentorGraph.read_schedule('C56E_formatted', 'mentor')
    # past_matches = [val for val in old_matches if val[0]<=start_slot]
    # m_matches_old = pmg.get_mentor_centered_view(old_matches)
    # # m_matches_2pod = dict((key, val) for key, val in m_matches_old.items() if len(val)==2)

    # # new graph with max_pod_per_mentor as 2
    # pmg = PodMentorGraph(pod_info, mentor_info, max_mentor_per_pod=1, max_pod_per_mentor=1,
    #                      mentor_requests=mentor_requests, affinity=None, start_slot=start_slot)
    # # for mentor_email, _matches in m_matches_2pod.items(): # deactivate 2-pod matches
    # #     m_idx = pmg.mentor_info['email'].index(mentor_email)
    # #     u = pmg.vertices.index(('mentor', m_idx))
    # #     pmg.capacity[u, pmg.vertices.index('sink')] = 0
    # #     for _, p_idx in _matches:
    # #         v = pmg.vertices.index(('pod', p_idx))
    # #         pmg.capacity[pmg.vertices.index('source'), v] = 0
    # for _, pod_name, mentor_email in past_matches:
    #     v = pmg.vertices.index(('pod', pmg.pod_info['name'].index(pod_name)))
    #     pmg.capacity[pmg.vertices.index('source'), v] = 0
    #     u = pmg.vertices.index(('mentor', pmg.mentor_info['email'].index(mentor_email)))
    #     pmg.capacity[u, pmg.vertices.index('sink')] -= 1
    # pmg.residual = pmg.capacity.copy()
    # pmg.load_matches(old_matches, volatility=-1000.)
    # tic = time.time()
    # matches_1 = pmg.get_matches()
    # toc = time.time()
    # print('{:d} min {:.1f} sec elapsed'.format(int((toc-tic)//60), (toc-tic)%60))
    # # for mentor_email, _matches in m_matches_2pod.items():
    # #     for s_idx, p_idx in _matches:
    # #         matches.append((s_idx, pmg.pod_info['name'][p_idx], mentor_email))
    # for p_m in past_matches:
    #     if p_m not in matches_1:
    #         matches_1.append(p_m)

    # pmg = PodMentorGraph(pod_info, mentor_info, max_mentor_per_pod=1, max_pod_per_mentor=2,
    #                       mentor_requests=mentor_requests, affinity=None, start_slot=start_slot)
    # for _, pod_name, mentor_email in past_matches:
    #     v = pmg.vertices.index(('pod', pmg.pod_info['name'].index(pod_name)))
    #     pmg.capacity[pmg.vertices.index('source'), v] = 0
    #     u = pmg.vertices.index(('mentor', pmg.mentor_info['email'].index(mentor_email)))
    #     pmg.capacity[u, pmg.vertices.index('sink')] -= 1
    # pmg.residual = pmg.capacity.copy()
    # pmg.load_matches(old_matches, volatility=-100.)
    # pmg.load_matches(matches_1, volatility=-1000.)
    # tic = time.time()
    # matches_2 = pmg.get_matches()
    # toc = time.time()
    # for p_m in past_matches:
    #     if p_m not in matches_2:
    #         matches_2.append(p_m)

    # matches = matches_2
    # r_id = random_id()
    # pmg.export_pod_schedule(r_id, matches)
    # pmg.export_mentor_schedule(r_id, matches)

    # m_matches_new = pmg.get_mentor_centered_view(matches)
    # with open('change.log.txt', 'w') as f:
    #     f.write('Change Log\n')
    #     for mentor_email in pmg.mentor_info['email']:
    #         if mentor_email not in m_matches_old:
    #             m_idx = pmg.mentor_info['email'].index(mentor_email)
    #             f.write('\n{} {}, {}\n'.format(
    #                 pmg.mentor_info['first_name'][m_idx],
    #                 pmg.mentor_info['last_name'][m_idx],
    #                 mentor_email
    #                 ))
    #             f.write('new mentor added\n')
    #             continue
    #         if set(m_matches_new[mentor_email])!=set(m_matches_old[mentor_email]):
    #             m_idx = pmg.mentor_info['email'].index(mentor_email)
    #             f.write('\n{} {}, {}, #pod {} --> {}\n'.format(
    #                 pmg.mentor_info['first_name'][m_idx],
    #                 pmg.mentor_info['last_name'][m_idx],
    #                 mentor_email,
    #                 len(set(m_matches_old[mentor_email])),
    #                 len(set(m_matches_new[mentor_email])),
    #                 ))
    #             to_remove = set(m_matches_old[mentor_email]).difference(set(m_matches_new[mentor_email]))
    #             to_add = set(m_matches_new[mentor_email]).difference(set(m_matches_old[mentor_email]))
    #             if to_remove:
    #                 f.write('remove\n')
    #                 for s_idx, p_idx in to_remove:
    #                     s_idx += 2 # convert from UTC to UTC+1
    #                     d_idx = s_idx//48
    #                     s_idx = s_idx%48

    #                     f.write('{} {}, {}\n'.format(
    #                         pmg._get_day_str(d_idx), pmg._get_slot_str(s_idx),
    #                         pmg.pod_info['name'][p_idx],
    #                         ))
    #             if to_add:
    #                 f.write('add\n')
    #                 for s_idx, p_idx in to_add:
    #                     s_idx += 2 # convert from UTC to UTC+1
    #                     d_idx = s_idx//48
    #                     s_idx = s_idx%48

    #                     f.write('{} {}, {}\n'.format(
    #                         pmg._get_day_str(d_idx), pmg._get_slot_str(s_idx),
    #                         pmg.pod_info['name'][p_idx],
    #                         ))
