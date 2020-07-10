# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   args
    Description: 
    Author:      wzj
    Date:        2019/7/30
-------------------------------------------------
    Change Activity:
-------------------------------------------------

"""
import logging

import base64
import ujson

from tornado.escape import url_escape, url_unescape

def deauth(auth: str):
    if not auth:
        return {}
    return ujson.loads(base64.b64decode(url_unescape(auth)))

def enauth(mcid, **kwargs):
    args = {
        'mcid': mcid
    }
    args.update(kwargs)
    temp = base64.b64encode(ujson.dumps(args).encode('utf-8'))
    return url_escape(temp.decode('utf-8'))

def enauth_other(**kwargs):
    args = {}
    args.update(kwargs)
    temp = base64.b64encode(ujson.dumps(args).encode('utf-8'))
    return url_escape(temp.decode('utf-8'))




# 异或算法
xorStr = lambda ss,cc: ''.join(chr(ord(s)^ord(cc)) for s in ss)
# 方法

def enauth_mcid(d_map, brandCode):

    """数据加密"""
    enStr = ujson.dumps(d_map)
    for cc in brandCode:
        enStr = xorStr(enStr,cc)
    if isinstance(enStr, str):
        enStr = enStr.encode("utf-8")
    return url_escape(base64.b64encode(enStr))

def deauth_mcid(mcid, key):
    """数据解密"""
    logging.error(mcid)
    data = url_unescape(mcid)
    deStr = base64.b64decode(data)
    deKey = key[::-1]
    if isinstance(deStr, bytes):
        deStr = deStr.decode("utf-8")
    for cc in deKey:
        deStr = xorStr(deStr,cc)
    logging.error(deStr)
    logging.error(key)
    return ujson.loads(deStr)



