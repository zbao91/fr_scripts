# -*- coding: utf-8 -*-

from core.application import url
from .image_processing import *

imp_urls = [
    url('/img_processing/im_cropped', ImageAligned, name='截图人脸'),
    url('/img_processing/cal_embd', CalFaceEmbd, name='计算特征向量'),
    url('/img_processing/load_embd', LoadFaceEmbd, name='读取特征向量文件'),
    url('/img_processing/data_split', DatasetSplit, name='将数据分类trian和test'),
    url('/img_processing/face_group', FaceGroup, name='将类似的人脸放到一起'),
    url('/img_processing/delete_single', DeleteSingle, name='将只有一张照片的人删掉'),
    url('/img_processing/extract_image', ExtractImage, name='每个人提取一张图片'),
]
