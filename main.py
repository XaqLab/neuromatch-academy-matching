# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

import argparse, pickle, time

from nmamatcher.utils import load_pod_info, load_mentor_info, load_student_abstracts
from nmamatcher.utils import random_id
from nmamatcher.maxflow_matcher import PodMentorGraph
from nmamatcher.maxflow_matcher import pod_mentor_topic_affinity, pod_mentor_dset_affinity

parser = argparse.ArgumentParser()
parser.add_argument('--affinity_type', default='dataset', choices=['abstract', 'dataset'])
args = parser.parse_args()

if __name__=='__main__':
    pod_info = load_pod_info('pod.map.csv')
    mentor_info = load_mentor_info('mentors.info.xlsx')

    if args.affinity_type=='abstract':
        student_abstracts = load_student_abstracts('student.abstracts.csv')
        affinity = pod_mentor_topic_affinity(pod_info, mentor_info, student_abstracts) # stochastic
    if args.affinity_type=='dataset':
        affinity = pod_mentor_dset_affinity(pod_info, mentor_info)

    special_list = {
        'athena.akrami@ucl.ac.uk': [3],
        'annaja@uw.edu': [4],
        'cchandr1@bu.edu': [3],
        'cohenm@pitt.edu': [2],
        'jdiedric@uwo.ca': [4],
        'raphael.s.m.kaplan@ntnu.no': [3],
        # 'pinotsis@mit.edu': [4],
        }
    pmg = PodMentorGraph(pod_info, mentor_info, max_pod_per_mentor=3, affinity=affinity, special_list=special_list)
    pmg.load_mentor_schedule('mentor.schedule_D6C2.csv', rigidity=5000)
    tic = time.time()
    r_id = random_id()
    pmg.export_schedules(r_id)
    toc = time.time()
    print('{:.1f} mins elapsed'.format((toc-tic)/60))
    with open(f'pod-mentor.graph_{r_id}.pkl', 'wb') as f:
        pickle.dump({
            'affinity_type': args.affinity_type,
            'model': pmg,
            }, f)
