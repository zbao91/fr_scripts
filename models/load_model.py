# -*- coding: utf-8 -*-
import torch
import os
import pickle as pkl
from torchvision import datasets
from torch.utils.data import DataLoader
from keras.models import load_model

from .facenet.inception_resnet_v1 import InceptionResnetV1
from .mtcnn.mtcnn import MTCNN
from .CSRNet.csrnet import CSRNet
from config.config import facebank, embeddings_path, liveDet_path, device, crowdCount_path

class ModelLoader(object):
    """
        模型读取器
    """
    def __init__(self):
        pass

    # 读取：人脸检测模型
    def load_mtcnn_model(self):
        return MTCNN(image_size=160,
                                margin=0,
                                min_face_size=20,
                                thresholds=[0.6, 0.7, 0.7],
                                factor=0.709,
                                post_process=True,
                                device=device)

    def load_facenet_model(self):
        return InceptionResnetV1(pretrained='vggface2').to(device).eval()

    # 读取：活体检测
    def load_live_detection_model(self):
        print('--->[活体检测] 开始导入活体检测模型')
        return load_model(os.path.join(liveDet_path, 'fas.h5'))

    # 读取：人脸库人脸特征和人脸名称
    def load_facebank(self):
        # 如果facebank的emb的文件是存在的，那么直接调用，如果不存在重新生成
        embd_file_path = os.path.join(embeddings_path, 'embd.pth')
        name_file_path = os.path.join(embeddings_path, 'name.pkl')
        def collate_fn(x):
            return x[0]
        # 检查文件是否已经存在，如果存在的话，直接从文件中读取，否则的话，在重新进行embeddings计算
        print('--->[人脸库] 开始初始化facebank中的embeddings')
        if not os.path.isfile(embd_file_path):
            print('--->[人脸库] 生成embedding文件')
            dataset = datasets.ImageFolder(facebank)
            dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
            loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=8)
            # todo: 多张图片最好有average embeddings
            aligned = []
            names = []
            count = 0
            for x, y in loader:
                print('facebank 生成 当前进度:%s'%count)
                count += 1
                x_aligned, prob = self.mtcnnModel(x, return_prob=True)
                if x_aligned is not None:
                    aligned.append(x_aligned)
                    names.append(dataset.idx_to_class[y])

            step = 5
            cur = 0
            facebank_embd = torch.empty(0, 512)
            while cur < len(aligned):
                tmp_aligned = aligned[cur: cur+step]
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
                facebank_embd_dict[names[i]] = torch.stack([facebank_embd_dict[names[i]], facebank_embd[i].float()], dim=0)

        for name in facebank_embd_dict:
            len_im = len(facebank_embd_dict[name].shape)
            deno = 1 if len_im == 1 else int(facebank_embd_dict[name].shape[0])
            facebank_embd_dict[name] = facebank_embd_dict[name] if len_im == 1 else facebank_embd_dict[name].sum(dim=0) / deno
            if name == 'angelina_jolie':
                print(facebank_embd_dict[name])
                print(deno)
        return facebank_embd_dict

    # 读取：人流量统计模型
    def load_crowd_counting(self):
        """
            读取点人头模型
        """
        print('--->[人流统计]开始读取人流统计模型CSRNet')
        model = CSRNet()
        model_path = os.path.join(crowdCount_path, 'model.pt')
        weights = torch.load(model_path, map_location='cpu') if device == torch.device('cpu') else torch.load(model_path)
        model.load_state_dict(weights)
        print('--->[人流统计]模型读取完毕')
        return model.eval()


