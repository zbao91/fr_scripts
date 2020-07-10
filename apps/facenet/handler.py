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
        data_path = ''
        datasets = os.listdir(data_path)
        #
        for dataset in datasets:
            val_path = os.path.join(data_path, dataset, 'val.pth')
            val_path = os.path.join(data_path, dataset, 'val.pth')
            for i in dataset:
                k = 10




        facebank_embd = 11
        loader_obj = ModelLoader()
        model = loader_obj.load_facenet_model()
        # 加载dict of embeddings
        torch.load()














        self.simplewrite()
        return

