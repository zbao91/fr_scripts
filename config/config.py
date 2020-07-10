# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name: config
    Description: 
    Author: zhiqi bao
    Date: 2020/07/02
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""
import torch
# DEBUG
DEBUG = True

# 模型所在目录
# 人脸识别
facebank = '/Users/zhiqibao/Desktop/Work_Wasu/server/facebank' # 人脸库图片所在目录
embeddings_path = '/Users/zhiqibao/Desktop/Work_Wasu/server/model_files/facebank' # 人脸特征信息文件所在目录
liveDet_path = '/Users/zhiqibao/Desktop/Work_Wasu/server/model_files/live_detection' # 活体检测模型所在目录
crowdCount_path = '/Users/zhiqibao/Desktop/Work_Wasu/server/models/CSRNet/checkpoint'

# device - cpu or gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


