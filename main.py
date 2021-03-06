# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

import argparse, time

from nmamatcher.utils import load_pod_info, load_mentor_info, load_mentor_requests, load_student_abstracts
from nmamatcher.utils import random_id
from nmamatcher.utils import create_fake_group
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

    print('initializing graph model...')

    # # reschedule based on old schedule, comment it out if you want to match
    # # from scratch
    # now_idx = 138 # day 3 slot 42, Wed 21:00 UTC
    # old_id = 'C56E'
    # old_matches_past, old_matches_future = PodMentorGraph.read_schedule(old_id, now_idx=now_idx)
    # pmg = PodMentorGraph(pod_info, mentor_info,
    #                      min_mentor_per_pod=1, max_mentor_per_pod=3,
    #                      min_pod_per_mentor=0, max_pod_per_mentor=3,
    #                      mentor_requests=mentor_requests,
    #                      fixed_matches=old_matches_past, now_idx=now_idx)
    # pmg.mark_matches(old_matches_future)

    # # # match pod and mentor from scratch, comment it out if you want to modify
    # # # an existing schedule
    # # pmg = PodMentorGraph(pod_info, mentor_info,
    # #                      min_mentor_per_pod=1, max_mentor_per_pod=3,
    # #                      min_pod_per_mentor=0, max_pod_per_mentor=3,
    # #                      mentor_requests=mentor_requests)

    # tic = time.time()
    # matches = pmg.get_matches()
    # toc = time.time()
    # print('{:d} min {:.1f} secs elapsed'.format(int((toc-tic)//60), (toc-tic)%60))
    # r_id = random_id()
    # pmg.export_pod_schedule(r_id, matches)
    # pmg.export_mentor_schedule(r_id, matches, print_idle=False)
    # pmg.export_changelog(old_id+'-'+r_id, old_matches_past+old_matches_future,
    #                      matches)

    group_info = create_fake_group(pod_info, max_group_size=5)


    # get a match without extra slots
    pmg = PodMentorGraph(group_info, mentor_info, stage=2,
                         min_mentor_per_pod=0, max_mentor_per_pod=1,
                         min_pod_per_mentor=0, max_pod_per_mentor=3,
                         mentor_requests=mentor_requests)
    tic = time.time()
    matches = pmg.get_matches()
    toc = time.time()
    print('{:d} min {:.1f} secs elapsed'.format(int((toc-tic)//60), (toc-tic)%60))

    # get a match with 1 extra slots
    pmg = PodMentorGraph(group_info, mentor_info, stage=2,
                         min_mentor_per_pod=0, max_mentor_per_pod=1,
                         min_pod_per_mentor=0, max_pod_per_mentor=3,
                         mentor_requests=mentor_requests, extra_slot=1)
    pmg.mark_matches(matches) # mark matches from above, optional
    tic = time.time()
    matches = pmg.get_matches()
    toc = time.time()

    r_id = random_id()
    print(f'\n{r_id}')
    pmg.export_pod_schedule(r_id, matches)
    pmg.export_mentor_schedule(r_id, matches)
