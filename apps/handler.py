# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   handler
    Description: 
    Date:        2019/7/23
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""

import logging
import time
import torch
import numpy as np
import datetime
import collections
import cv2
import io

from torchvision import transforms
from tornado.web import HTTPError
from tornado.web import RequestHandler
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from copy import deepcopy, copy
from matplotlib import cm

# from models import facenetModel, mtcnnModel, liveDetModel, facebankEmbd
from config.config import DEBUG, device
from .tools import BatchTimer, Logger

class BaseHandler(RequestHandler):
    """
    handler基类
    """
    def __init__(self, application, request, **kwargs):
        # inherit self and applicaiton
        super().__init__(application, request, **kwargs)
        super(BaseHandler, self).__init__(application, request, **kwargs)

    # 接受上传的图片
    def receive_im(self):
        files = self.request.files
        valid_keys = ['im', 'img']
        ims = []
        for k in files.keys():
            if k in valid_keys:
                for i in range(len(files[k])):
                    im = files[k][i]['body']
                    im = Image.open(io.BytesIO(im))
                    ims.append(im)
        return ims

    def clear_write(self):
        self._write_buffer = []

    def write(self, chunk):
        self.clear_write()
        if isinstance(chunk, dict):
            self.return_msg = chunk.get('msg', '无提示')
            self.return_code = chunk.get('code', -100)
        if self.get_status() == 200:
            super().write(self.formate_datatime(chunk))
        else:
            super().write(chunk)
        if DEBUG:
            logging.error(chunk)

    def simplewrite(self, code=1, msg='成功', data=None, **kwargs):
        res = {
            'code': code,
            'msg': msg,
            'time': int(time.time()),
            'data': data
        }
        # res['data'] = {} if data is None else data
        res.update(kwargs)
        self.write(res)

    def formate_datatime(self, rd, formate="%m-%d %H:%M"):
        data = copy(rd)
        if isinstance(rd, dict):
            for key, value in rd.copy().items():

                if isinstance(value, collections.Mapping):
                    data[key] = self.formate_datatime(value, formate)
                if isinstance(value, datetime.datetime):
                    data[key] = self.datetime_to_string(value, formate)
                if isinstance(value, list):
                    data[key] = [self.formate_datatime(item) for item in value]
        if isinstance(rd, list):
            data = [self.formate_datatime(item) for item in rd]
        return data

    def datetime_to_string(self,dt, formate="%m-%d %H:%M"):
        c_time = dt.strftime(formate)
        return c_time

class FacenetHandler(RequestHandler):
    """
        Facenet模型通用Handler
    """
    def __init__(self, application, request, **kwargs):
        # inherit self and applicaiton
        super().__init__(application, request, **kwargs)
        super(BaseHandler, self).__init__(application, request, **kwargs)

    def eval(self, model, eval_dataset, loss_fn, batch_metrics={'time': BatchTimer()}, writer=SummaryWriter()):
        """
            数据验证
        """
        writer.iteration, writer.interval = 0, 10
        logger = Logger('eval', length=len(eval_dataset), calculate_mean=True)
        loss = 0
        metrics = {}
        for i_batch, (x, y) in enumerate(eval_dataset):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss_batch = loss_fn(y_pred, y)

            metrics_batch = {}
            for metric_name, metric_fn in batch_metrics.items():
                metrics_batch[metric_name] = metric_fn(y_pred, y).detach().cpu()  # detach() to avoid copy
                metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

            loss_batch = loss_batch.detach().cpu()
            loss += loss_batch
            logger(loss, metrics, i_batch)

        loss = loss / (i_batch + 1)
        metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}
        if writer is not None and not model.training:
            writer.add_scalars('loss', {'Eval': loss.detach()}, writer.iteration)
            for metric_name, metric in metrics.items():
                writer.add_scalars(metric_name, {'Eval': metric})
        return


