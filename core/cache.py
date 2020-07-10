# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   lru_cache
    Description: 
    Author:      wzj
    Date:        2019/8/5
-------------------------------------------------
    Change Activity:
        copy from functool
        修改： 增加缓存时间限制
-------------------------------------------------
"""

import ujson
import time
from _thread import RLock
from dbdriver import rdb

def _make_key(args, user_function):
    fn = str(user_function)
    return fn + '_'.join(str(arg) for arg in args)

def lru_cache(maxsize=128, timeout=60*60*2):
    def decorating_function(user_function):
        wrapper = _lru_cache_wrapper(user_function, maxsize, timeout)
        return wrapper
    return decorating_function

def _lru_cache_wrapper(user_function, maxsize, timeout):
    # Constants shared by all lru cache instances:
    sentinel = object()          # unique object used to signal cache misses
    make_key = _make_key         # build a key from the function arguments
    PREV, NEXT, KEY, RESULT, TIMEOUT = 0, 1, 2, 3, 4   # names for the link fields

    cache = {}
    timeout = timeout
    hits = misses = 0
    full = False  # 是否超大小
    cache_get = cache.get    # bound method to lookup a key or return None
    cache_len = cache.__len__  # get cache size without calling len()
    lock = RLock()           # because linkedlist updates aren't threadsafe
    root = []                # root of the circular doubly linked list
    root[:] = [root, root, None, None, None]     # initialize by pointing to self

    if maxsize == 0:

        def wrapper(*args, **kwds):
            # No caching -- just a statistics update
            nonlocal misses
            misses += 1
            result = user_function(*args, **kwds)
            return result

    elif maxsize is None:

        def wrapper(*args, **kwds):
            # Simple caching without ordering or size limit
            nonlocal hits, misses
            key = make_key(args)
            result = cache_get(key, sentinel)
            if result is not sentinel:
                hits += 1
                return result
            misses += 1
            result = user_function(*args, **kwds)
            cache[key] = result
            return result

    else:

        def wrapper(self, *args, **kwds):
            # Size limited caching that tracks accesses by recency
            nonlocal root, hits, misses, full, timeout
            key = make_key(args, user_function)
            cts = int(time.time())
            with lock:
                link = cache_get(key)
                if link is not None:
                    # Move the link to the front of the circular queue
                    link_prev, link_next, _key, result, intime = link
                    if cts - intime > timeout:
                        # 数据超时, 需重新获取
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        del cache[_key]
                        full = (cache_len() >= maxsize)
                    else:
                        link_prev[NEXT] = link_next
                        link_next[PREV] = link_prev
                        last = root[PREV]
                        last[NEXT] = root[PREV] = link
                        link[PREV] = last
                        link[NEXT] = root
                        hits += 1
                        return result
                misses += 1
            result = user_function(self, *args, **kwds)
            with lock:
                if key in cache:
                    # Getting here means that this same key was added to the
                    # cache while the lock was released.  Since the link
                    # update is already done, we need only return the
                    # computed result and update the count of misses.
                    pass
                elif full:
                    # Use the old root to store the new key and result.
                    oldroot = root
                    oldroot[KEY] = key
                    oldroot[RESULT] = result
                    oldroot[TIMEOUT] = timeout
                    # Empty the oldest link and make it the new root.
                    # Keep a reference to the old key and old result to
                    # prevent their ref counts from going to zero during the
                    # update. That will prevent potentially arbitrary object
                    # clean-up code (i.e. __del__) from running while we're
                    # still adjusting the links.
                    root = oldroot[NEXT]
                    oldkey = root[KEY]
                    oldresult = root[RESULT]
                    root[KEY] = root[RESULT] = None
                    # Now update the cache dictionary.
                    del cache[oldkey]
                    # Save the potentially reentrant cache[key] assignment
                    # for last, after the root and links have been put in
                    # a consistent state.
                    cache[key] = oldroot
                else:
                    # Put result in a new link at the front of the queue.
                    last = root[PREV]
                    link = [last, root, key, result, int(time.time())]
                    last[NEXT] = root[PREV] = cache[key] = link
                    # Use the cache_len bound method instead of the len() function
                    # which could potentially be wrapped in an lru_cache itself.
                    full = (cache_len() >= maxsize)
            return result

    return wrapper


def set_to_redis(key, value, expire=2*60*60):
    if isinstance(value, str):
        rdb.set(key, value)
    else:
        rdb.set(key, ujson.dumps(value))
    rdb.expire(key, expire)

def get_from_redis(key, json_fmt=False):
    v = rdb.get(key)
    if v is None:
        return None
    else:
        if json_fmt:
            return ujson.loads(v)
        else:
            return v
