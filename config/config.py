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
import os
# DEBUG
DEBUG = True

# 模型所在目录
# 人脸识别
project_path = os.path.abspath(os.path.join(os.getcwd(), '..')) # base path of project directory
facebank = '/Users/zhiqibao/Desktop/Work_Wasu/server/model_files/facebank' # 人脸特征信息文件所在目录
embeddings_path = '/Users/zhiqibao/Desktop/Work_Wasu/server/model_files/facebank' # 人脸特征信息文件所在目录
liveDet_path = '/Users/zhiqibao/Desktop/Work_Wasu/server/model_files/live_detection' # 活体检测模型所在目录
crowdCount_path = '/Users/zhiqibao/Desktop/Work_Wasu/server/models/CSRNet/checkpoint'

project = 'face_recognition'
sub_project ='scripts'

# 备份路径
backup_path = os.path.join(project_path, 'backup')

# device - cpu or gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


