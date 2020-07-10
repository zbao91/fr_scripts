# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   db
    Description: 
    Author:      wzj
    Date:        2019/8/1
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""

import ujson
from dbdriver import rdb


def get_from_redis(key, defalut=None):
    return rdb.get(key) or defalut

def set_to_redis(key, value, expire=2*60*60):
    redis_cli = rdb
    if isinstance(value, str):
        redis_cli.set(key, value)
    else:
        redis_cli.set(key, ujson.dumps(value))
    redis_cli.expire(key, expire)
