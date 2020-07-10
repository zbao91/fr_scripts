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
import copy

from torchvision import datasets, transforms
from distutils.dir_util import copy_tree, remove_tree
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from shutil import copyfile, copy2

from ..handler import BaseHandler
from models import ModelLoader
from ..tools import fixed_image_standardization, CompareByEmbd
from config.config import device
from utils.common import create_name, rename_old_dir, create_dir

class ImageAligned(BaseHandler):
    def get(self):
        """
            使用mtcnn截图图片人头
        """
        data_dir = '/home/huasu/Desktop/facenet/files/data/Asian'
        error_dir = '/home/huasu/Desktop/facenet/files/data/photos_error'
        bakup_dir = '/home/huasu/Data/asian'

        # 读取模型
        loader_obj = ModelLoader()
        model = loader_obj.load_mtcnn_model()
        dirs = os.listdir(data_dir)

        # 获取数据，如果打不开图片，则
        length = len(dirs)
        for idx, dir in enumerate(dirs):
            if dir == '.DS_Store':
                continue
            if idx % 100 == 0:
                print(idx, '/', length)
            cur_dir = data_dir + '/%s' % dir
            files = os.listdir(cur_dir)
            if len(files) == 0:
                remove_tree(cur_dir)
            has_error = 0
            error_file_list = []
            for file in files:
                try:
                    img = Image.open(cur_dir + '/' + file)
                except:
                    has_error += 1
            # 如果图片有问题，丢到error文件夹内
            if has_error > 0:
                copy_tree(cur_dir, error_dir + '/' + dir)
                print('error:', cur_dir)
                remove_tree(cur_dir)

        # perform mtcnn facial detection
        dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
        dataset.samples = [
            (p, p.replace(data_dir, data_dir + '_cropped'))
            for p, _ in dataset.samples
        ]
        loader = DataLoader(
            dataset,
            num_workers=8,
            batch_size=5,
            collate_fn=self.collate_pil
        )
        names = []
        for i, (x, y) in enumerate(loader):
            # 截图图片后
            try:
                model(x, save_path=y)
            except:
                logging.info('test')
                continue
            # 将文件夹保存在bakup中
            for k in y:
                k = '/'.join(k.split('/')[-2:])
                name = k.split('/')[-2]
                names.append(name)

                if not os.path.isdir(bakup_dir):
                    os.mkdir(bakup_dir)

                if not os.path.isdir(bakup_dir + '/' + name):
                    os.mkdir(bakup_dir + '/' + name)

                shutil.move(data_dir + '/' + k, bakup_dir + '/' + name)
                # os.remove(data_dir + '/' + k)
            print('\rBatch {} of {}'.format(i + 1, len(loader)))

        for name in names:
            if os.path.isdir(data_dir + '/' + name):
                remove_tree(data_dir + '/' + name)
        return

    def collate_pil(self, x):
        out_x, out_y = [], []
        for xx, yy in x:
            out_x.append(xx)
            out_y.append(yy)
        return out_x, out_y

class DatasetSplit(BaseHandler):
    """
        将图片分为测试集和验证集
    """
    def get(self):
        data_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/Asian'
        dst_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/Asian_Split_2'
        method = 1
        # 方法1：每个人下面提取一张照片
        if method == 1:
            self.method1(data_path, dst_path)
        return

    def method1(self, data_path, dst_path, amount_dataset=5, num_im=1):
        """
            每个人提取一张照片作为验证照片
            amount_dataset: amount of dataset
            num_im: amount of image for validation
        """
        # 读取数据
        dataset = datasets.ImageFolder(data_path)
        path_split = data_path.split('/')
       # how many different dataset to generate
        for i in range(amount_dataset):
            print('当前数据组：%s'%i)
            # check directory for test and validation dataset
            sub_path = os.path.join(dst_path, 'dataset_%d'%i)
            val_path, test_path = os.path.join(sub_path, 'val'), os.path.join(sub_path, 'test')
            for tmp_path in [test_path, val_path]:
                if not os.path.isdir(tmp_path):
                    os.makedirs(tmp_path)
            # split data
            imgs = [i[0] for i in dataset.imgs]
            counter = {}
            np.random.shuffle(imgs)
            for f in imgs:
                name = f.split('/')[-2]
                f_name = '/'.join(f.split('/')[-2:])
                if not name in counter:
                    counter[name] = 1
                    val_or_test = 0 # 0: val, 1: test
                else:
                    if counter[name] < num_im:
                        val_or_test = 0
                    else:
                        val_or_test = 1
                    counter[name] += 1

                # 如果是test
                if val_or_test == 1:
                    sub_dst_path = os.path.join(test_path, f_name)
                else:
                    sub_dst_path = os.path.join(val_path, f_name)
                if not os.path.isdir(os.path.dirname(sub_dst_path)):
                    os.makedirs(os.path.dirname(sub_dst_path))
                copyfile(f, sub_dst_path)
        return

class CalFaceEmbd(BaseHandler):
    """
        计算脸的特征向量，只能对使用mtcnn处理过的图片使用
    """
    def get(self):
        method = 2
        facebank_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/facegroup_test'  # 人脸库地址
        embd_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/facegroup_embd'
        # 计算facebank的embeddings, 然后按照姓名进行归类
        if method == 1:
            self.method1(facebank_path, embd_path)
        # 获取目录下的所有照片，然后计算embeddings
        elif method == 2:
            dir_list = os.listdir(facebank_path)
            for _dir in dir_list:
                if _dir == '.DS_Store':
                    continue
                _dir = os.path.join(facebank_path, _dir)
                self.method2(_dir, embd_path)
        return

    def method1(self, facebank_path, base_path):
        """
            把所有的facebank（文档结构: name1/img1.jpg, name1/img2.jpg）中的
        """
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        # backup old directory and create new directory
        rename_old_dir(base_path)
        new_dir_name = '%s_current' % (create_name(base_path, 'embd'))
        create_dir(os.path.join(base_path, new_dir_name))
        embd_name = os.path.basename(facebank_path)
        embd_path = os.path.join(base_path, new_dir_name, '%s.pth'%embd_name)  # 特征向量地址

        # 读取模型, 可以根据地址修改模型
        ml_obj = ModelLoader()
        facenet = ml_obj.load_facenet_model()

        def collate_pil(x):
            out_x, out_y = [], []
            for xx, yy in x:
                out_x.append(xx)
                out_y.append(yy)
            return out_x, out_y

        # 读取人脸库的数据，然后生成embeddings
        transf = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization

        ])
        dataset = datasets.ImageFolder(facebank_path, transform=transf)
        name_dict = dict((dataset.class_to_idx[i], i) for i in dataset.class_to_idx)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        val_dataset = DataLoader(
            dataset,
            num_workers=4,
            batch_size=4,
            collate_fn=collate_pil
        )
        embd_dict = {}  # key: name, value: list of embd
        for idex, (x, y) in enumerate(val_dataset):
            input_x = torch.stack(x).to(device)
            pred_y = facenet(input_x).detach().cpu()
            for idx in range(len(pred_y)):
                name = name_dict.get(y[idx])
                embd = pred_y[idx]
                if not name in embd_dict:
                    embd_dict[name] = [embd]
                else:
                    embd_dict[name].append(embd)
        torch.save(embd_dict, embd_path)
        return

    def method2(self, facebank_path, embd_path):
        """
         获取路径下所有的图片，然后转化成tensor，然后计算embeddings, 并保存
        """
        files = os.listdir(facebank_path)
        loader_obj = ModelLoader()
        model = loader_obj.load_facenet_model()
        tmp_dict = {}
        for f in files:
            if f == '.DS_Store':
                continue
            f_path = os.path.join(facebank_path, f)
            im = Image.open(f_path)
            im = transforms.ToTensor()(im).unsqueeze_(0)
            dist = model(im).detach().cpu()
            tmp_dict[f] = dist
        embd_name = '%s.pth'%(os.path.basename(facebank_path))
        if not os.path.isdir(embd_path):
            os.makedirs(embd_path)
        embd_path = os.path.join(embd_path, embd_name)
        torch.save(tmp_dict, embd_path)
        return

    def method3(self, facebank_path, base_path):
        """
            和method1类似，只不过把所有的数据都放到
        """
        if not os.path.isdir(base_path):
            os.makedirs(base_path)
        # backup old directory and create new directory
        rename_old_dir(base_path)
        new_dir_name = '%s_current' % (create_name(base_path, 'embd'))
        create_dir(os.path.join(base_path, new_dir_name))
        embd_path = os.path.join(base_path, new_dir_name, 'embd_dict.pth')  # 特征向量地址

        # 读取模型, 可以根据地址修改模型
        ml_obj = ModelLoader()
        facenet = ml_obj.load_facenet_model()

        def collate_pil(x):
            out_x, out_y = [], []
            for xx, yy in x:
                out_x.append(xx)
                out_y.append(yy)
            return out_x, out_y

        # 读取人脸库的数据，然后生成embeddings
        transf = transforms.Compose([
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization

        ])
        dataset = datasets.ImageFolder(facebank_path, transform=transf)
        name_dict = dict((dataset.class_to_idx[i], i) for i in dataset.class_to_idx)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        val_dataset = DataLoader(
            dataset,
            num_workers=4,
            batch_size=4,
            collate_fn=collate_pil
        )
        embd_dict = {}  # key: name, value: list of embd
        for idex, (x, y) in enumerate(val_dataset):
            input_x = torch.stack(x).to(device)
            pred_y = facenet(input_x).detach().cpu()
            for idx in range(len(pred_y)):
                name = name_dict.get(y[idx])
                embd = pred_y[idx]
                if not name in embd_dict:
                    embd_dict[name] = [embd]
                else:
                    embd_dict[name].append(embd)
        torch.save(embd_dict, embd_path)
        return


class FaceGroup(BaseHandler):
    """
        将每天的人脸进行归纳，将相似的人脸放在一起
    """
    def get(self):
        base_embd_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/facegroup_embd'
        org_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/facegroup_test'
        dst_path = '/Users/zhiqibao/Desktop/Work_Wasu/人脸识别/face_data/facegrouped_test'
        embds = os.listdir(base_embd_path)
        for embd_name in embds:
            if embd_name.startswith('.'):
                continue
            embd_path = os.path.join(base_embd_path, embd_name)
            embd_data = torch.load(embd_path) # key为文件名称
            embd_2nd = copy.deepcopy(embd_data)
            name_idx = 0
            grouped_im = []
            unique_list = []
            for f_name in embd_data:
                # 如果这张图片已经被整理过了，那么直接跳过
                if f_name in grouped_im:
                    continue
                embd_val = embd_data[f_name]
                # 剔除掉图片本身
                embd_2nd.pop(f_name)
                same_list = []
                for f_name1 in embd_2nd:
                    embd_val1 = embd_2nd[f_name1]
                    # 拿到两个embeddings，然后进行比较，如果小于0.8那么就将两张图片放到一起
                    dist = CompareByEmbd([embd_val, embd_val1])
                    # 如果数据小于0.7, 认为他们是相同的，然后把
                    if float(dist) < 0.7:
                        same_list.append(f_name1)

                for same_im in same_list:
                    embd_2nd.pop(same_im)
                # 然后将图片放到一起
                ## 创建文件夹, 如果有存在相同的情况
                if len(same_list) > 0:
                    same_list.append(f_name)
                else:
                    unique_list.append(f_name)

                if len(same_list) > 0:
                    grouped_path = os.path.join(dst_path, embd_name.split('.')[0], str(name_idx))
                    if not os.path.isdir(grouped_path):
                        os.makedirs(grouped_path)
                    name_idx += 1
                    for im in same_list:
                        im_dst_path = os.path.join(grouped_path, im)
                        im_org_path = os.path.join(org_path, embd_name.split('.')[0], im)
                        copyfile(im_org_path, im_dst_path)
                # 记录哪些图片是已经被分过类的
                grouped_im.extend(same_list)

            # 将没有分类的图片放到公共目录下
            for fname in unique_list:
                im_org_path = os.path.join(org_path, embd_name.split('.')[0], fname)
                im_dst_path = os.path.join(dst_path, embd_name.split('.')[0], fname)
                copyfile(im_org_path, im_dst_path)
        return

