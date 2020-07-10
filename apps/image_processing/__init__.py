# -*- coding: utf-8 -*-

from core.application import url
from .image_processing import *

imp_urls = [
    url('/im_processing/im_cropped', ImageAligned, name='截图人脸'),
    url('/im_processing/cal_embd', CalFaceEmbd, name='计算特征向量'),
    url('/im_processing/data_split', DatasetSplit, name='将数据分类trian和test'),
    url('/im_processing/face_group', FaceGroup, name='将类似的人脸放到一起'),
]
