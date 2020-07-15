# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

import argparse, time

from nmamatcher.utils import load_pod_info, load_mentor_info, load_student_abstracts
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

    if args.affinity_type=='abstract':
        student_abstracts = load_student_abstracts('student.abstracts.csv')
        affinity = pod_mentor_topic_affinity(pod_info, mentor_info, student_abstracts) # stochastic
    if args.affinity_type=='dataset':
        affinity = pod_mentor_dset_affinity(pod_info, mentor_info)

    pmg = PodMentorGraph(pod_info, mentor_info, max_pod_per_mentor=2)
    pmg.load_matches(pmg.read_schedule('C56E', 'mentor'), volatility=1.)
    r_id = random_id()
    tic = time.time()
    matches = pmg.get_matches()
    pmg.export_pod_schedule(r_id, matches)
    pmg.export_mentor_schedule(r_id, matches)
    toc = time.time()
    print('{:d} min {:.1f} sec elapsed'.format(int((toc-tic)//60), (toc-tic)%60))
