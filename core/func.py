# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   func
    Description: 
    Author:      wzj
    Date:        2019/8/2
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""
import ujson
import re
import logging
import time
import random
import hashlib
import base64

from dbdriver import rdb
from urllib.parse import quote

html_re = re.compile(r'<.*?b.*?>')

xorStr = lambda ss,cc: ''.join(chr(ord(s)^ord(cc)) for s in ss)

def get_brandCode_logo(brandCode):
    return "https://007vin.com/img/{brandCode}.png".format(brandCode=brandCode)


def get_brandCode_name(brandCode):
    return brandCode

 # 1 红色(对应color_value = 0)
def get_is_filter_by_colorvalue(single_dict):
    return 0 if single_dict.get('colorvalue', 1) else 1

def get_bool_filter(single_dict, is_filter):
    return (not is_filter) or (is_filter and single_dict.get('colorvalue', 1))

def get_bool_filter_test(c, is_filter):
    return (not is_filter) or (is_filter and c)

def remove_html_br(origin_str):
    return html_re.sub('', origin_str)

def cache_func_with_args(func, args, key, expire, rand_range=0):
    redis_cli = rdb
    v = redis_cli.get(key)
    if not v:
        expire = random.randint(expire - rand_range, expire + rand_range)
        v = func(*args)
        redis_cli.set(key, ujson.dumps(v))
        redis_cli.expire(key, expire)
        return v
    else:
        logging.error(v)
        return ujson.loads(v)

def cache_func(func, key, expire, rand_range=0):
    redis_cli = rdb
    v = redis_cli.get(key)
    if not v:
        expire = random.randint(expire - rand_range, expire + rand_range)
        v = func()
        logging.error(v)
        redis_cli.set(key, ujson.dumps(v))
        redis_cli.expire(key, expire)
        return v
    else:
        logging.error(v)
        return ujson.loads(v)


def calculate_md5(key:str):
    m = hashlib.md5()
    m.update(bytes(key.encode()))
    return m.hexdigest().lower()

# def calculate_token(yc_id, username):
#     key = "%s%s%s%s" % (yc_id, str(time.time() * 1000), yc_id, APPKEY)
# hashid = self.getMd5("%s%s%s%s" % (yc_id, str(time.time() * 1000), yc_id, appKey))

def encrypt_auth(data,key='ulei'):
    """数据加密"""
    enStr = data
    for cc in key:
        enStr = xorStr(enStr,cc)
    if isinstance(enStr, str):
        enStr = enStr.encode("utf-8")
    return quote(base64.b64encode(enStr))





if __name__ == '__main__':
    a = [
        (1, 0, True),
        (0, 0, True),
        (1, 1, True),
        (0, 1, False)
    ]
    for colorvalue, is_filter, ret in a:
        print(ret == get_bool_filter_test(colorvalue, is_filter))
