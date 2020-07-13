# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:25:03 2020

@author: Zhe
"""

import pandas
import numpy as np
from collections import Counter

import re
import string
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer


SLOT_NUM = 48

GROUP_SLOTS = {} # NMA time zone groups in UTC
offsets = {'1': 0, '2': 14, '3': 34}
hour_spans = {
    'A': np.arange(-8, 0),
    'B': np.concatenate((np.arange(-4, 0), np.arange(14, 18))),
    'C': np.arange(14, 22),
    }
for key_a in offsets:
    for key_b in hour_spans:
        GROUP_SLOTS[key_a+key_b] = (hour_spans[key_b]+offsets[key_a])


def load_pod_info(pod_csv):
    r"""Loads pod information.

    Args
    ----
    pod_csv: str
        The csv file renamed from ``'pod_maps-Grid view.csv'``.

    Returns
    -------
    pod_info: dict
        A dictionary containing information of all pods.

        `'pod_num'`: int
            The number of pods.
        `'name'`: list of str
            The name of each pod.
        `'pod_email'`: list of str
            The e-mail address of each pod.
        `'timezone_label'`: list of str
            The time zone label of each pod, e.g. ``'1A'``, ``'3B'``.
        `'slots'`: list
            The time slots for project of each pod, determined by
            `'timezone_label'`. Each element is a list of ints, containing
            indices ranging in :math:`[0, 48)` corresponding to the 48
            half-hour slots in UTC.
        `'student_emails'`: list
            Each element is a list of e-mail addresses of students within this
            pod.

    """
    df = pandas.read_csv(pod_csv)
    pod_names_ = np.array(df['pod_name'].tolist(), dtype=np.object)
    pod_info = {'name': np.unique(pod_names_).tolist()}
    pod_info['pod_num'] = len(pod_info['name'])

    pod_info.update({
        'pod_email': [],
        'timezone_label': [],
        'slots': [],
        'student_emails': [],
        })
    for p_name in pod_info['name']:
        idxs, = (pod_names_==p_name).nonzero()
        pod_info['pod_email'].append(df['pod_email'][idxs[0]].lower())
        pod_info['timezone_label'].append(df['pod_slot'][idxs[0]])
        pod_info['slots'].append(GROUP_SLOTS[df['pod_slot'][idxs[0]]])
        pod_info['student_emails'].append([
            df['student_email'][idx].lower() for idx in idxs
            ])
    return pod_info


def load_mentor_info(mentor_xlsx):
    r"""Loads mentor hour availability.

    Time slots and the available weekdays are all in UTC.

    Args
    ----
    mentor_xlsx: str
        The xlsx file downloaded from the `Google Sheet <https://docs.google.com/spreadsheets/d/1I7qbfcmtCNU8vPCMFxum2CZRl-nwNVDPlG-j9_kYevw>`_.

    Returns
    -------
    mentor_info: dict
        A dictionary containing available time of all mentors.

        `'mentor_num'`: int
            The number of mentors.
        `'email'`: list of str
            The e-mail address of each mentor.
        `'first_name'`, `'last_name'`: list of str
            The first name and last name of each mentor.
        `'timezone'`: list of str
            The local time zone of each mentor.
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
    mentor_info = {
        'email':[val.lower() for val in df['Q33'].tolist()[2:]],
        'first_name': df['Q24'].tolist()[2:],
        'last_name': df['Q2'].tolist()[2:],
        'timezone': df['Q5'].tolist()[2:],
        }
    has_duplicate = False
    for m_email, count in Counter(mentor_info['email']).items():
        if count>1:
            print(f'{m_email} occurred {count} times')
            has_duplicate = True
    if has_duplicate:
        print(f'duplicate e-mails found, please fix {mentor_xlsx}')

    mentor_info.update({
        'mentor_num': len(mentor_info['email']),
        'primary_days': [],
        'primary_slots': [],
        'flexibility': [],
        'secondary_days': [],
        'secondary_slots': [],
        })

    def get_slots(start_time, duration_str):
        slot_start = start_time.hour*2
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
        mentor_info['primary_days'].append([
            d_idx for d_idx in range(15) if isinstance(
                df['Q3_{}_{}'.format(d_idx//5+1, d_idx%5+1)][i], str
                )
            ])
        mentor_info['primary_slots'].append(get_slots(df['Q25'][i], df['Q41'][2]))

        if df['Q42'][i]=='No':
            f_i = 0
        elif np.isnan(df['Q47_1'][i]):
            f_i = 0.9 # default preference for second choice
        else:
            f_i = df['Q47_1'][i]/10
        mentor_info['flexibility'].append(f_i)

        mentor_info['secondary_days'].append([
            d_idx for d_idx in range(15) if isinstance(
                df['Q44_{}_{}'.format(d_idx//5+1, d_idx%5+1)][i], str
                )
            ])
        mentor_info['secondary_slots'].append(
            [] if f_i==0 else get_slots(df['Q45'][i], df['Q46'][i])
            )

    mentor_info['abstracts'] = []
    df = pandas.read_excel(mentor_xlsx, 'Confirmed Mentors - Short Googl')
    emails_ = [val.lower() for val in df['Email Address']]
    assert len(set(emails_))==len(emails_), 'duplicate e-mails found in abstracts sheet'
    for m_idx, email in enumerate(mentor_info['email']):
        if email in emails_:
            mentor_info['abstracts'].append(
                df[df.columns[11]][emails_.index(email)]
                )
        else:
            mentor_info['abstracts'].append('')
    return mentor_info


def load_student_abstracts(student_csv):
    r"""Loads student abstracts.

    Args
    ----
    student_csv: str
        The csv file renamed from ``Filtered view.csv``.

    Returns
    -------
    student_abstracts: dict
        A dictionary containing abstracts of all students.

        `'student_num'`: int
            The number of students.
        `'email'`: list of str
            The e-mail address of each student.
        `'abstracts'`: list of str
            The abstract of each student. Obviously invalid ones are replaced
            with an empty str `''`.

    """
    df = pandas.read_csv(student_csv)
    assert len(np.unique(df['id']))==len(df)
    student_abstracts = {
        'student_num': len(df),
        'email': [val.lower() for val in df['id'].tolist()],
        'abstracts': df['abstracts'].tolist(),
        }
    for i, a in enumerate(student_abstracts['abstracts']):
        if not isinstance(a, str) or len(a)<100:
            student_abstracts['abstracts'][i] = ''
    return student_abstracts


def slot_label(s_idx):
    r"""Returns slot label.

    Args
    ----
    s_idx: int
        The index of half-hour time slots, starting from 12:00 AM with 0.

    Returns
    -------
    s_label: str
        A string for the time slot. `'(+1)'` or `'(-1)'` may be used for day
        change.

    """
    assert s_idx>=-48 and s_idx<96
    if s_idx<0:
        s_label = ' (-1)'
        s_idx += 48
    elif s_idx>=48:
        s_label = ' (+1)'
        s_idx -= 48
    else:
        s_label = ''
    def hour_label(h_idx):
        h_label = ' AM' if h_idx<24 or h_idx==48 else ' PM'
        h_idx %= 24
        h_label = '{:2d}:{:02d}'.format(
            ((h_idx//2)-1)%12+1, (h_idx%2)*30,
            )+h_label
        return h_label
    s_label = '{} - {}'.format(
        hour_label(s_idx), hour_label(s_idx+1)
        )+s_label
    return s_label


stemmer = PorterStemmer()
w_tokenizer = WhitespaceTokenizer()
punct_re = re.compile('[{}]'.format(re.escape(string.punctuation)))

def preprocess(text, stemming=True):
    r"""Applies Snowball stemmer to string.

    Adapted from `<https://github.com/titipata/paper-reviewer-matcher>`_.

    Args
    ----
    text : str
        Input abstract.
    stemming : bool
        Porter stemmer is applied if ``True``.
    """

    if isinstance(text, (type(None), float)):
        text_preprocess = ''
    else:
        text = unidecode(text).lower()
        text = punct_re.sub(' ', text) # remove punctuation
        if stemming:
            text_preprocess = [stemmer.stem(token) for token in w_tokenizer.tokenize(text)]
        else:
            text_preprocess = w_tokenizer.tokenize(text)
        text_preprocess = ' '.join(text_preprocess)
    return text_preprocess
