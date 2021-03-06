# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:25:03 2020

@author: Zhe
"""

import pandas, random
import numpy as np
from collections import Counter

import re, string
from unidecode import unidecode
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WhitespaceTokenizer


SLOT_NUM = 48

GROUP_SLOTS = {} # NMA time zone groups in UTC
offsets = {'1': 0, '2': 14, '3': 34}
hour_spans = {
    'A': np.arange(-8, 0),
    'B-': np.arange(-4, 0),
    'B': np.concatenate((np.arange(-4, 0), np.arange(14, 18))),
    'B+': np.arange(14, 18),
    'C': np.arange(14, 22),
    }
for key_a in offsets:
    for key_b in hour_spans:
        GROUP_SLOTS[key_a+key_b] = (hour_spans[key_b]+offsets[key_a])

POD_DSET_OPTIONS = {
    'Single Unit': 'SingleUnit',
    'FMRI': 'fMRI',
    'EEG': 'EEG',
    }
MENTOR_DSET_OPTIONS = [
    'EEG/ECoG/MEG/LFP', 'Spikes/Calcium', 'MRI/fMRI/BulkCalcium',
    'Behavior(Choice)', 'Behavior(Motor/HighDim)'
    ]
DSET_AFFINITY = {
    'SingleUnit': [0.8, 1., 0.6, 0.2, 0.4],
    'fMRI': [0.8, 0.6, 1., 0.8, 0.2],
    'EEG': [1., 0.6, 0.8, 0.8, 0.4],
    }


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
        `'tz_group'`: list of str
            The time zone label of each pod, e.g. ``'1A'``, ``'3B'``.
        `'dset_option'`: list of str
            The dataset option of each pod, reformatted through
            ``POD_DSET_OPTIONS``.
        `'student_emails'`: list
            Each element is a list of e-mail addresses of students within this
            pod.

    """
    df = pandas.read_csv(pod_csv)
    pod_names_ = np.array(df['pod_name'].tolist(), np.object)
    pod_info = {'name': np.unique(pod_names_).tolist()}
    pod_info['pod_num'] = len(pod_info['name'])

    pod_info.update({
        'pod_email': [],
        'tz_group': [],
        'dset_option': [],
        'student_emails': [],
        })
    for p_name in pod_info['name']:
        idxs, = (pod_names_==p_name).nonzero()
        pod_info['pod_email'].append(df['pod_email'][idxs[0]].lower())
        pod_info['tz_group'].append(df['pod_slot'][idxs[0]])
        pod_info['dset_option'].append(POD_DSET_OPTIONS[df['pod_dataset'][idxs[0]]])
        pod_info['student_emails'].append([
            df['student_email'][idx].lower() for idx in idxs
            ])
    return pod_info


def create_fake_group(pod_info, max_group_size=5):
    group_info = {
        'name': [],
        'tz_group': [],
        'dset_option': [],
        }
    for p_idx in range(pod_info['pod_num']):
        g_size = (len(pod_info['student_emails'][p_idx])-1)//max_group_size+1
        for g_idx in range(g_size):
            group_info['name'].append(pod_info['name'][p_idx]+'_{}'.format(g_idx))
            group_info['tz_group'].append(pod_info['tz_group'][p_idx])
            group_info['dset_option'].append(pod_info['dset_option'][p_idx])
    group_info['pod_num'] = len(group_info['name']) # use pod keys
    return group_info


def load_mentor_info(mentor_xlsx):
    r"""Loads mentor information.

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
            corresponding to the 48 half-hour slots. Time slots and the
            available days are all in UTC.
        `'flexibility'`: list of floats
            The willingness for second choice of each mentor. Each element is
            a float in :math:`[0, 1]`.
        `'secondary_days'`: list
            The second choice of available days of each mentor, with the same
            format as `'primary_days'`.
        `'secondary_slots'`: list
            The second choice of time slots of each mentor, with the same
            format as `'secondary_days'`.
        `'abstract'`: list of str
            The abstract of each mentor, extracted from sheet ``'Confirmed
            Mentors - Short Googl'``.
        `'dset_option'`: list
            The preferred dataset option to work with of each mentor. Each
            element is a list containing values from ``MENTOR_DSET_OPTIONS``.

    """
    def check_duplicate_email(df_series):
        r"""Returns a list of emails if no duplicate is found."""
        emails = [val.lower() for val in df_series]
        if len(set(emails))!=len(emails):
            for m_email, count in Counter(emails).items():
                if count>1:
                    print(f'{m_email} occurred {count} times')
            raise RuntimeError('duplicate e-mails found')
        return emails

    def get_slots(start_time, duration_str):
        r"""Returns the specified slot indices.

        Args
        ----
        start_time: datetime
            A time object returned by pandas.
        duration_str: str
            The answer from Google sheet, only 4 different values are
            encountered.

        Returns
        -------
        A list of valid slot indices.

        """
        slot_start = start_time.hour*2+start_time.minute//30
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

    df = pandas.read_excel(mentor_xlsx, 'Project mentors - Final hours')
    mentor_info = {
        'email': check_duplicate_email(df['Q33'][2:]),
        'first_name': df['Q24'].tolist()[2:],
        'last_name': df['Q2'].tolist()[2:],
        'timezone': df['Q5'].tolist()[2:],
        }

    mentor_info.update({
        'mentor_num': len(mentor_info['email']),
        'primary_days': [],
        'primary_slots': [],
        'flexibility': [],
        'secondary_days': [],
        'secondary_slots': [],
        })

    for i in range(2, len(df)):
        mentor_info['primary_days'].append([
            d_idx for d_idx in range(15) if isinstance(
                df['Q3_{}_{}'.format(d_idx//5+1, d_idx%5+1)][i], str
                )
            ])
        mentor_info['primary_slots'].append(get_slots(df['Q25'][i], df['Q41'][i]))

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

    mentor_info['abstract'] = []
    df = pandas.read_excel(mentor_xlsx, 'Confirmed Mentors - Short Googl')
    emails = check_duplicate_email(df['Email Address'])
    for m_email in mentor_info['email']:
        if m_email in emails:
            mentor_info['abstract'].append(
                df[df.columns[11]][emails.index(m_email)]
                )
        else:
            mentor_info['abstract'].append('')

    mentor_info['dset_option'] = []
    df = pandas.read_excel(mentor_xlsx, 'Confirmed Mentors - Full Applic')
    emails = check_duplicate_email(df['Q33'][2:])
    keys = ['Q29_1', 'Q29_2', 'Q29_3', 'Q29_5', 'Q29_6']
    for m_email in mentor_info['email']:
        if m_email in emails:
            idx = emails.index(m_email)+2
            mentor_info['dset_option'].append([
                label for label, key in zip(MENTOR_DSET_OPTIONS, keys) \
                if isinstance(df[key][idx], str)
             ])
        else:
            mentor_info['dset_option'].append([])
    return mentor_info


def load_mentor_requests(request_xlsx):
    r"""Loads mentor requests.

    Args
    ----
    request_xlsx: str
        The xlsx file with mentor requests.

    Returns
    -------
    mentor_requests: dict
        A dictionary containing mentor requests.

        `'request_num'`: int
            The number of requests.
        `'email'`: str
            The mentor e-mail address.
        `'type'`: str
            The type of each request, can be ``'deactivate'``, ``'add'`` and
            ``'remove'``.
        `'d_idx'`: float
            The day index for the request, only valid when `'type'` value is
            ``'add'`` or ``'remove'``, should be in :math:`[0, 15)`.
        `'s_idx'`: float
            The slot index for the request, only valid when `'type'` value is
            ``'add'`` or ``'remove'``, should be in :math:`[0, SLOT_NUM)`.
            Time slots specified are in UTC.

    """
    df = pandas.read_excel(request_xlsx)
    mentor_requests = {
        'request_num': len(df),
        'email': [val.lower() for val in df['email'].tolist()],
        'type': df['type'].tolist(),
        'd_idx': df['day'].tolist(),
        's_idx': df['slot'].tolist(),
        }
    return mentor_requests


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

    Returns
    -------
    Preprocessed abstract string.

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


def random_id(str_len=4):
    r"""Returns a random ID string.

    Args
    ----
    str_len: int
        Desired string length.

    Returns
    -------
    A string of specified length, containing ``'0'``-``'9'``, ``'A'``-``'F'``.

    """
    return ''.join(['{:X}'.format(random.randrange(16)) for _ in range(str_len)])
