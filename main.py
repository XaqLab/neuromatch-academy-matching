# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

from nmamatcher.utils import load_pod_info, load_mentor_info, load_student_abstracts
from nmamatcher.maxflow_matcher import PodMentorGraph
from nmamatcher.maxflow_matcher import pod_mentor_topic_affinity, pod_mentor_dset_affinity

if __name__=='__main__':
    pod_info = load_pod_info('pod.maps.csv')
    mentor_info = load_mentor_info('mentors.info.xlsx')

    # student_abstracts = load_student_abstracts('student.abstracts.csv')
    # affinity = pod_mentor_topic_affinity(pod_info, mentor_info, student_abstracts) # stochastic

    affinity = pod_mentor_dset_affinity(pod_info, mentor_info)

    pmg = PodMentorGraph(pod_info, mentor_info, affinity=affinity)
    pmg.export_mentor_schedule('mentor.schedule.csv')
