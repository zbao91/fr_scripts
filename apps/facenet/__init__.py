# -*- coding: utf-8 -*-

from core.application import url
from .FaceNet import *

facenet_urls = [
    url('/facenet/performance', Performance, name='模型准确率测试'),
    url('/facenet/finetune', FineTune, name='模型调参'),

]
