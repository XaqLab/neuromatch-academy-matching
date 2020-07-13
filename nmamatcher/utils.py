# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:25:03 2020

@author: Zhe
"""

import pandas
import numpy as np
from collections import Counter


SLOT_NUM = 48

PROJECT_HOURS = {} # NMA time zone groups in UTC
offsets = {'1': 0, '2': 14, '3': 34}
hour_spans = {
    'A': np.arange(-8, 0),
    'B': np.concatenate((np.arange(-4, 0), np.arange(14, 18))),
    'C': np.arange(14, 22),
    }
for key_a in offsets:
    for key_b in hour_spans:
        PROJECT_HOURS[key_a+key_b] = (hour_spans[key_b]+offsets[key_a])%SLOT_NUM


def load_mentor_hours(mentor_xlsx):
    r"""Loads mentor hour availability.

    Time slots and the available weekdays are all in UTC.

    Args
    ----
    mentor_xlsx: str
        The xlsx file downloaded from the `Google Sheet <https://docs.google.com/spreadsheets/d/1I7qbfcmtCNU8vPCMFxum2CZRl-nwNVDPlG-j9_kYevw>`_.

    Returns
    -------
    mentor_hours: dict
        A dictionary containing available time of all mentors.

        `'mentor_num'`: int
            The number of mentors.
        `'email'`: list of str
            E-mail address of each mentor.
        `'first_name'`, `'last_name'`: list of str
            First name and last name of each mentor.
        `'timezone'`: list of str
            Local time zone of each mentor.
        `'primary_days'`: list
            The first choice of available days of each mentor. Each element is
            a list of ints, containing indices for the 15 weekdays over the 3
            weeks.
        `'primary_slots'`: list
            The first choice of time slots of each mentor. Each element is a
            list of ints, containing indices ranging in :math:`[0, 48)`
            corresponding to the 48 half-hour slots.
        `'flexibility'`: list of floats
            The willingness for second choice of each mentor. Each element is
            a float in :math:`[0, 1]`.
        `'secondary_days'`: list
            The second choice of available days of each mentor, with the same
            format as `'primary_days'`.
        `'secondary_slots'`: list
            The second choice of time slots of each mentor, with the same
            format as `'secondary_days'`.

    """
    df = pandas.read_excel(mentor_xlsx, 'Project mentors - Final hours')
    mentor_hours = {
        'email':[val.lower() for val in df['Q33'].tolist()[2:]],
        'first_name': df['Q24'].tolist()[2:],
        'last_name': df['Q2'].tolist()[2:],
        'timezone': df['Q5'].tolist()[2:],
        }
    has_duplicate = False
    for m_email, count in Counter(mentor_hours['email']).items():
        if count>1:
            print(f'{m_email} occurred {count} times')
            has_duplicate = True
    if has_duplicate:
        raise RuntimeWarning(f'duplicate e-mails found, please fix {mentor_xlsx}')

    mentor_hours.update({
        'mentor_num': len(mentor_hours['email']),
        'primary_days': [],
        'primary_slots': [],
        'flexibility': [],
        'secondary_days': [],
        'secondary_slots': [],
        })

    def get_slots(start_time, duration_str):
        slot_start = start_time.hour*2+2 # convert from UTC+0 to UTC+1
        if duration_str=='1/2 hour':
            slot_end = slot_start+1
        elif duration_str=='1 hour':
            slot_end = slot_start+2
        elif duration_str=='1.5 hour':
            slot_end = slot_start+3
        elif duration_str=='2 hour':
            slot_end = slot_start+4
        else:
            raise RuntimeError(f'duration \'{duration_str}\' unrecognized')
        return list(range(slot_start, slot_end))

    for i in range(2, len(df)):
        mentor_hours['primary_days'].append([
            d_idx for d_idx in range(15) if isinstance(
                df['Q3_{}_{}'.format(d_idx//5+1, d_idx%5+1)][i], str
                )
            ])
        mentor_hours['primary_slots'].append(get_slots(df['Q25'][i], df['Q41'][2]))
        mentor_hours['flexibility'].append(
            0 if np.isnan(df['Q47_1'][i]) else df['Q47_1'][i]/10
            )
        mentor_hours['secondary_days'].append([
            d_idx for d_idx in range(15) if isinstance(
                df['Q44_{}_{}'.format(d_idx//5+1, d_idx%5+1)][i], str
                )
            ])
        mentor_hours['secondary_slots'].append(
            [] if np.isnan(df['Q47_1'][i]) else get_slots(df['Q45'][i], df['Q46'][2])
            )
    return mentor_hours
