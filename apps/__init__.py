# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name: __init__.py
    Description:
    Author: zhiqi bao
    Date: 2019/7/23
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""

from .image_processing import imp_urls
from .facenet import facenet_urls

urls = []
urls.extend(imp_urls)
urls.extend(facenet_urls)

