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
import argparse
import datetime
import csv

from torchvision import datasets, transforms
from distutils.dir_util import copy_tree, remove_tree
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from ..handler import BaseHandler
from ..tools import fixed_image_standardization, Flatten, normalize, Image_Processing
from config.config import device, project_path
from models import ModelLoader
from models.facenet.inception_resnet_v1 import InceptionResnetV1
from utils.common import create_dir, create_name, log_recording

from models.facenet.utils import training

category = 'facenet_related'

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
        im = transforms.ToTensor()(im).unsqueeze_(0).to(device)
        predict = model(im).detach().cpu()


        return

class FineTune(BaseHandler):
    def parse_args(self):
        """
        Get arguments from input
        """
        parser = argparse.ArgumentParser(description='Finetune pretrained facenet')
        parser.add_argument('--lr', default=0.01, help='learning rate of model')
        parser.add_argument('--type', default='train', help='which process for this script')
        parser.add_argument('--tag', type=int, default=1, help='finetune id')
        parser.add_argument('--train-amount', type=int, default=20000,
                            help='how many data will be used to finetune pretrianed model')
        parser.add_argument('--train-ratio', type=float, default=0.7, help='train/test ratio')
        parser.add_argument('--nGPU', type=int, default=torch.cuda.device_count(), help='number of gpu')
        parser.add_argument('--batch-size', type=int, default=4, help='batch size')
        parser.add_argument('--epochs', type=int, default=500, help='epoches')
        parser.add_argument('--freeze-layer', default=1, help='which type of layer to freeze')
        parser.add_argument('--decay', default=[30, 200], help='which type of layer to freeze')
        parser.add_argument('--source', default='auto_machine', help='where source from')
        parser.add_argument('--workers', default=0 if os.name == 'nt' else 8, help='amount of worker')
        parser.add_argument('--dist_amount', default=50, help='how many pairs of image be used to calculate distance')
        args = parser.parse_args()
        return args

    def get(self):
        args = self.parse_args()
        current_date = datetime.date.today().strftime('%Y%m%d')

        # init finetune files and record config
        args.img_path = os.path.join(project_path, 'data', args.source, current_date)
        args.base_path = os.path.join(project_path, category, 'finetune')
        create_dir(args.base_path)
        tmp_finetune_path = create_name(args.base_path, current_date, is_datetime=False)
        args.finetune_path = os.path.join(args.base_path, tmp_finetune_path)
        create_dir(args.finetune_path)

        # load images
        train_loader, test_loader = self.load_sample(args)

        # load model
        model = InceptionResnetV1(
            classify=True,
            pretrained='vggface2',
            num_classes=args.num_class
        ).to(device)

        # calcualte distance between different images
        args.cmpt_info = self.samples_for_dist_compute(test_loader, args)

        # record parameters
        config_info = 'learning_rate: {}, freeze_layer: {}, decay: {}, epoches: {}'.format(args.lr, args.freeze_layer, args.decay, args.epochs)
        config_path = os.path.join(args.finetune_path, 'config.txt')
        log_recording(config_path, config_info, 'w+')

        # start to train
        self.logfile_update(args)
        self.train_model(model, train_loader, test_loader, args)

        return

    # correct image path
    def load_sample(self, args):
        trans = transforms.Compose([
            transforms.Resize((int(288 * (256 / 224)), int(288 * (256 / 224)))),
            np.float32,
            transforms.ToTensor(),
            fixed_image_standardization,
        ])
        # load imags
        dataset_all = datasets.ImageFolder(args.img_path, transform=trans)
        # split image to train and test set
        img_inds = np.arange(len(dataset_all))
        img_inds = img_inds[:args.train_amount]
        dataset_size = len(img_inds)
        split = int(np.floor(args.train_ratio * dataset_size))
        tmp_dict = dataset_all.class_to_idx
        idx_to_class = dict((v, k) for k, v in tmp_dict.items())
        np.random.shuffle(img_inds)
        train_indices, val_indices = img_inds[:split], img_inds[split:]

        # get train and test loader
        train_loader = DataLoader(
            dataset_all,
            num_workers=args.workers,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(train_indices),
        )
        test_loader = DataLoader(
            dataset_all,
            num_workers=args.workers,
            batch_size=args.batch_size,
            sampler=SubsetRandomSampler(val_indices),
        )
        class_list = set([])
        for tmp_loader in [train_loader, test_loader]:
            for idx, (x, y) in enumerate(tmp_loader):
                for tmp_class in y:
                    if not tmp_class in class_list:
                        class_list.add(tmp_class.item())
        args.num_class = len(class_list)
        return train_loader, test_loader

    def samples_for_dist_compute(self, _dataset, args):
        # define dist images
        cmpt_info = []
        imgs_path = [i[0] for i in _dataset.dataset.samples]  #
        # get same comparison
        tmp_dict = {}
        for tmp_img in imgs_path:
            tmp_name = tmp_img.split('/')[-2]
            if not tmp_name in tmp_dict:
                tmp_dict[tmp_name] = [tmp_img]
            else:
                tmp_dict[tmp_name].append(tmp_img)
        idx = 0
        for tmp_name in tmp_dict:
            if idx >= args.dist_amount:
                break
            new_dict = {
                'status': 'same',
                'imgs': tmp_dict[tmp_name]
            }
            cmpt_info.append(new_dict)
            idx += 1

        np.random.shuffle(imgs_path)
        idx = 0
        for idx, v in enumerate(imgs_path):
            if imgs_path[idx] != imgs_path[-idx]:
                new_dict = {
                    'status': 'diff',
                    'imgs': [imgs_path[idx], imgs_path[-idx]]
                }
                cmpt_info.append(new_dict)
                idx += 1
            if idx >= args.dist_amount:
                break
        return cmpt_info

    def train_model(self, model, train_dataset, test_dataset, args):
        """
        train model
        """
        # freeze layer and get new model
        model = self.model_layer_freeze(model, args)

        # define optimizer, scheduler, dataset and dataloader
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = MultiStepLR(optimizer, args.decay)
        loss_fn = torch.nn.CrossEntropyLoss()

        # define loss function
        metrics = {
            'fps': training.BatchTimer(),
            'acc': training.accuracy
        }
        writer = SummaryWriter()
        writer.iteration, writer.interval = 0, 10

        # define log files and checkpoints
        model_prefix = create_name(args.finetune_path, 'ckpt')

        # start to train
        highest_acc = 0
        for epoch in range(args.epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, args.epochs))
            print('-' * 10)
            args.cur_epoch = epoch

            # compute distance between images
            self.dist_compute(model, args)

            # start to train model
            model.train()
            args.status = 'train'
            loss_train, vmets_train = training.pass_epoch(
                model, loss_fn, train_dataset, optimizer, scheduler,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer, args=args
            )

            # start to evaluate model
            model.eval()
            loss_test, vmets_test = training.pass_epoch(
                model, loss_fn, test_dataset, optimizer, scheduler,
                batch_metrics=metrics, show_running=True, device=device,
                writer=writer, args=args
            )
            # check whether the log file is ext
            if os.path.isfile(args.train_res_path):
                log_ext = 1
            else:
                log_ext = 0

            # record train result
            with open(args.train_res_path, 'a+') as fd:
                csv_writer = csv.writer(fd)
                if log_ext == 0:
                    csv_writer.writerow(["epoch", "train acc", "train loss", "test acc", "test loss"])
                csv_writer.writerow(
                    [epoch, vmets_train.get('acc', 0).item(), loss_train.item(), vmets_test.get('acc', 0).item(),
                     loss_test.item()])

            cur_acc = float(vmets_test.get('acc'))
            if epoch == 1:
                highest_acc = cur_acc
            else:
                if highest_acc < cur_acc:
                    highest_acc = cur_acc
                    # save model
                    model_name = '%s_%s_%.4f.pt' % (model_prefix, epoch, cur_acc)
                    model_pth = os.path.join(args.finetune_path, model_name)
                    torch.save({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optim': optimizer.state_dict(),
                        'vaccu': 0
                    }, model_pth)

        writer.close()
        return

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

    def logfile_update(self, args):
        # record log to files
        args.train_res_path = os.path.join(args.finetune_path, 'train_res.csv')
        args.dist_res_same_path = os.path.join(args.finetune_path, 'dist_res_same.csv')
        args.dist_res_diff_path = os.path.join(args.finetune_path, 'dist_res_diff.csv')
        return

    def dist_compute(self, model, args):
        """
        compute distance of paired images
        model: cnn model
        args.cmpt_info:
            * type: list of dict
            * key of dict:
                - status: same/diff
                - imgs: list of images
        """
        model.eval()
        img_obj = Image_Processing()
        for i in args.cmpt_info:
            status = i.get('status', '')  # same or diff
            imgs_path = i.get('imgs')
            names = list(set([k.split('/')[-2] for k in imgs_path]))
            imgs = img_obj.read_img(imgs_path)
            dist = self.compare(model, imgs)
            info = 'status: {} - epoch:{} - names:{} - distance:{}'.format(status, args.cur_epoch, '|'.join(names),
                                                                           dist)
            if status == 'same':
                cur_path = os.path.join(args.finetune_path, 'dist_res_same.csv')
            else:
                cur_path = os.path.join(args.finetune_path, 'dist_res_diff.csv')
            log_recording(cur_path, info, 'a+')
        return

    def compare(self, model, imgs):
        aligned = torch.stack(imgs).to(device)
        embeddings = model(aligned).detach().cpu()
        dists = (embeddings[0] - embeddings[1]).norm().item()
        return dists