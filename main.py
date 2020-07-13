# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

from nmamatcher.utils import load_pod_info, load_mentor_availability, load_student_abstracts
from nmamatcher.maxflow_matcher import PodMentorGraph

if __name__=='__main__':
    pod_info = load_pod_info('pod.maps.csv')
    mentor_availability = load_mentor_availability('mentors.info.xlsx')
    student_abstracts = load_student_abstracts('student.abstracts.csv')

    pmg = PodMentorGraph(pod_info, mentor_availability)
    pmg.export_mentor_schedule('mentor.schedule.csv')
