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

    tic = time.time()
    matches = pmg.get_matches()
    toc = time.time()
    print('{:d} min {:f} secs elapsed'.format(int((toc-tic)//60), (toc-tic)%60))
    r_id = random_id()
    pmg.export_pod_schedule(r_id, matches)
    pmg.export_mentor_schedule(r_id, matches)
    pmg.export_changelog(old_id+'-'+r_id, old_matches_past+old_matches_future,
                         matches)
