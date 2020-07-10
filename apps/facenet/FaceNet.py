# -*- coding: utf-8 -*-
"""
    图片处理
"""
import torchvision.transforms.functional as F
import os
import shutil
import logging
import torch
import pickle as pkl
import numpy as np

from torchvision import datasets, transforms
from distutils.dir_util import copy_tree, remove_tree
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler

from ..handler import BaseHandler
from ..tools import fixed_image_standardization
from config.config import device
from models import ModelLoader

class Performance(BaseHandler):
    def get(self):
        """
            模型准确度检测，直接使用embeddings就可以
        """
        data_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/asian_split_embeddings'
        datasets = os.listdir(data_path)
        groups_list = set([])
        for dataset in datasets:
            data_group = '_'.join(dataset.split('_')[:2])
            groups_list.add(data_group)

        for group in groups_list:
            val_path = os.path.join(data_path, '%s_test.pth'%group) #
            test_path = os.path.join(data_path, '%s_val.pth' % group)
            val_data = torch.load(val_path)
            test_data = torch.load(test_path)
            self.method_average(val_data, test_data)
            self.method_normal(val_data, test_data)
        self.simplewrite()
        print('finish')
        print('-' * 20)
        return

    def method_average(self, val_data, test_data):
        """
            使用每个人的平均embedding来预测身份
        """
        total_embds = []
        names = []
        for name in val_data:
            embds = val_data[name]
            embds_mean = torch.mean(torch.stack(embds).float(), dim=0).unsqueeze(0)
            total_embds.append(embds_mean)
            names.append(name)
        embds_mean = torch.cat(total_embds, dim=0)
        predicts = []
        y = []
        for name in test_data:
            test_embds = test_data[name][0].unsqueeze(0)
            dist = torch.norm(embds_mean - test_embds, dim=1)
            name_idx = torch.argmin(dist).item()
            predicts.append(names[name_idx])
            y.append(name)
        acc = (np.array(predicts) == np.array(y)).mean()
        len_data = len(y)
        print('Average - 准确率: %.3f, 数据长度：%d'%(acc, len_data))
        return

    def method_normal(self, val_data, test_data):
        """
            使用最普通的方法，直接和全部的进行比较
        """
        total_embds = []
        names = []
        for name in val_data:
            embds = val_data[name]
            np.random.shuffle(embds)
            total_embds.append(embds[0].unsqueeze(0))
            names.append(name)
        embds_all = torch.cat(total_embds, dim=0)
        predicts = []
        y = []
        for name in test_data:
            test_embds = test_data[name][0].unsqueeze(0)
            dist = torch.norm(embds_all - test_embds, dim=1)
            name_idx = torch.argmin(dist).item()
            predicts.append(names[name_idx])
            y.append(name)
        acc = (np.array(predicts) == np.array(y)).mean()
        len_data = len(y)
        print('Normal - 准确率: %.3f, 数据长度：%d' % (acc, len_data))
        return

class Output_test(BaseHandler):
    def get(self):
        loader_obj = ModelLoader()
        model = loader_obj.load_facenet_model()
        im = Image.open('/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/asian_cropped/000/000_4.bmp')
        im = transforms.ToTensor()(im).unsqueeze_(0)
        model(im)








        return



class FineTune(BaseHandler):
    def get(self):





        return

