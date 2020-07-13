# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:25:25 2020

@author: Zhe
"""

from nmamatcher.utils import load_pod_info, load_mentor_availability

if __name__=='__main__':
    pod_csv = 'pod.maps.csv'
    mentor_xlsx = 'mentors.info.xlsx'

    pod_info = load_pod_info(pod_csv)
    mentor_availability = load_mentor_availability(mentor_xlsx)
