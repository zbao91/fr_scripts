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
from torchvision import datasets, transforms
from distutils.dir_util import copy_tree, remove_tree
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler

from ..handler import BaseHandler
from models import ml_obj
from ..tools import Flatten, normalize

class FacenetEval(BaseHandler):
    def get(self):
        """
            测验facenet的准确率
        """
        facebank_path = '' # 人脸库地址
        embd_path = '' # 特征向量地址
        nfile_path = '' # 名字问价你地址
        # 读取模型
        model = ml_obj.load_facenet_model()

        # 如果facebank的emb的文件是存在的，那么直接调用，如果不存在重新生成
        embd_file_path = os.path.join(embd_path, 'embd.pth')
        name_file_path = os.path.join(nfile_path, 'name.pkl')

        def collate_fn(x):
            return x[0]

        # 检查文件是否已经存在，如果存在的话，直接从文件中读取，否则的话，在重新进行embeddings计算
        dataset = datasets.ImageFolder(facebank_path)
        dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=8)
        names = []
        aligned = []
        count = 0
        for x, y in loader:
            print('facebank embd 生成 当前进度:%s' % count)
            count += 1




            x_aligned, prob = self.mtcnnModel(x, return_prob=True)
            if x_aligned is not None:
                aligned.append(x_aligned)
                names.append(dataset.idx_to_class[y])

            step = 5
            cur = 0
            facebank_embd = torch.empty(0, 512)
            while cur < len(aligned):
                tmp_aligned = aligned[cur: cur + step]
                tmp_aligned = torch.stack(tmp_aligned).to(device)
                tmp_facebank_embd = self.facenetModel(tmp_aligned).detach().cpu()
                facebank_embd = torch.cat([facebank_embd, tmp_facebank_embd], dim=0)
                cur += step

            # save names file and facebank embeddings file
            torch.save(facebank_embd, embd_file_path)
            with open(name_file_path, "wb") as fout:
                pkl.dump(names, fout, protocol=pkl.HIGHEST_PROTOCOL)
        else:
            print('--->[人脸库] 读取 embedding文件')
            facebank_embd = torch.load(embd_file_path)
            with open(name_file_path, "rb") as fout:
                names = pkl.load(fout)
        facebank_embd_dict = {}
        for i in range(len(names)):
            if not names[i] in facebank_embd_dict:
                facebank_embd_dict[names[i]] = facebank_embd[i].float()
            else:
                facebank_embd_dict[names[i]] = torch.stack([facebank_embd_dict[names[i]], facebank_embd[i].float()],
                                                           dim=0)

        for name in facebank_embd_dict:
            len_im = len(facebank_embd_dict[name].shape)
            deno = 1 if len_im == 1 else int(facebank_embd_dict[name].shape[0])
            facebank_embd_dict[name] = facebank_embd_dict[name] if len_im == 1 else facebank_embd_dict[name].sum(
                dim=0) / deno
            if name == 'angelina_jolie':
                print(facebank_embd_dict[name])
                print(deno)



    def collate_pil(self, x):
        out_x, out_y = [], []
        for xx, yy in x:
            out_x.append(xx)
            out_y.append(yy)
        return out_x, out_y

    def model_layer_freeze(self, model, args):
        # freeze_layer: 1 - freeze all layers excpet fc7
        if args.freeze_layer == 1:
            len_params = len(list(model.parameters()))
            for idx, param in enumerate(model.parameters()):
                if idx < len_params - 5:
                    param.requires_grad = False

        # 2: # use model from https://towardsdatascience.com/finetune-a-facial-recognition-classifier-to-recognize-your-face-using-pytorch-d00a639d9a79
        elif args.freeze_layer == 2:
            layer_list = list(model.children())[-5:]  # all final layers
            model_ft = torch.nn.Sequential(*list(model.children())[:-5])
            for param in model_ft.parameters():
                param.requires_grad = False
            model_ft.avgpool_1a = torch.nn.AdaptiveAvgPool2d(output_size=1)
            model_ft.last_linear = torch.nn.Sequential(
                Flatten(),
                torch.nn.Linear(in_features=1792, out_features=512, bias=False),
                normalize()
            )
            model_ft.logits = torch.nn.Linear(layer_list[-1].in_features, args.num_class)
            model_ft.softmax = torch.nn.Softmax(dim=1)
            model = model_ft
        return model