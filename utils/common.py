# coding: utf-8
"""
some common tools
"""

import os
import datetime

from pathlib import Path
from shutil import move


def create_name(_path, prefix, is_datetime=True):
    """
        create name with prefix and datetime
    """
    # TODO: 2020/07/09: 将这个代码整合进通用代码里面
    current_date = datetime.date.today().strftime('%Y%m%d')
    print(current_date)
    base_name = '{prefix}_{date}'.format(prefix=prefix, date=current_date) if is_datetime else prefix
    dir_list = os.listdir(_path)
    idx = 0
    for _dir in dir_list:
        idx += 1 if base_name in _dir else 0
    new_name = '{}_{}'.format(base_name, idx)
    return new_name

def rename_old_dir(path):
    """
        rename dir with 'current'
    """
    dir_list = os.listdir(path)
    dir_names = [i for i in dir_list if 'current' in i]
    for dir_name in dir_names:
        org_path = os.path.join(path, dir_name)
        new_path = os.path.join(path, dir_name.replace('_current', ''))
        move(org_path, new_path)
    return

def create_dir(path):
    """
        create directory if not exist
    """
    if not os.path.isdir(path):
        os.makedirs(path)
    return

def log_recording(file_path, info, _type):
    """
    recording infos to specific path
    """
    with open(file_path, _type) as f:
        f.write('%s\n'%info)
    return


