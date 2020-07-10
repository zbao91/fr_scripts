# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   httpmixin
    Description: 远端http请求
    Author:      wzj
    Date:        2019/7/24
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""

import logging

from tornado.httpclient import HTTPRequest, AsyncHTTPClient, HTTPError
from tornado.httputil import url_concat, urlencode
from tornado.escape import json_decode

from config.rpc import USER_HOST, EPC_HOST, IMAGE_HOST, ACCOUNT_HOST
from core.errors import BaseRPCError, BaseError


class GetHTTPRequest(HTTPRequest):
    """
    praam: 查询参数 字典对象
    """
    def __init__(self, url, param=None, **kwargs):
        if param:
            url = url_concat(url, param)
        super().__init__(url, "GET", **kwargs)

class PostHTTPRequest(HTTPRequest):
    """
    仅支持简单的application/x-www-form-urlencoded类型的post请求
    data: None or dict
    """
    def __init__(self, url, data=None, param=None, **kwargs):
        body = data or {}
        body.update(kwargs.pop('body', {}))
        if param:
            url = url_concat(url, param)
        super().__init__(url, 'POST', body=urlencode(body), **kwargs)


class HTTPContent(object):
    """远程http请求

    如果发生异常，get/post会直接raise
    """
    host = ''

    @classmethod
    def check_callback(self, callback):
        if callback and not callable(callback):
            raise NotImplementedError
            # raise NError(str(callback)+'必须为可调用对象')


    @classmethod
    async def get(cls, uri:str, param=None, arg_rename=None, **kwargs):
        if not uri.startswith('http'):
            if not cls.host:
                raise NotImplementedError
            url = cls.host + uri
        else:
            url = uri
        param = {} if param is None else param
        param.update(kwargs)
        if arg_rename:
            for arg, narg in arg_rename.items():
                if arg in param:
                    param[narg] = param.pop(arg)
        request = GetHTTPRequest(url, param)
        return await cls.send_request(request)

    @classmethod
    async def post(cls, uri, data=None, param=None, arg_rename=None, **kwargs):
        if not uri.startswith('http'):
            if not cls.host:
                raise NotImplementedError
            url = cls.host + uri
        else:
            url = uri
        data = {} if data is None else data
        data.update(kwargs)
        if arg_rename:
            for arg, narg in arg_rename.items():
                if arg in data:
                    data[narg] = data.pop(arg)
        request = PostHTTPRequest(url, data=data, param=param)
        return await cls.send_request(request)

    @classmethod
    async def send_request(self, request):
        try:
            response = await AsyncHTTPClient().fetch(request)
            rjson = json_decode(response.body)
        except (HTTPError, ValueError) as e:
            raise BaseRPCError
        else:
            return rjson


class EPCHTTPContent(HTTPContent):

    host = EPC_HOST


class USERHTTPContent(HTTPContent):

    host = USER_HOST


class AccountHTTPContent(HTTPContent):

    host = ACCOUNT_HOST


class ImageHttpContent(HTTPContent):

    host = IMAGE_HOST

    uri = '/save/imgs/post'

    @classmethod
    async def post_image(self, filename=None, data=None, classfiy=None):
        """
        统一图片服务|上传接口 {
          'imginfo' : {
              'host' : 'https://phyimgs.007vin.com',
              'uri' : '/logo/CORTECO/CORTECO.bmp'
              "remoteurl" : "https://phyimgs.007vin.com/logo/CORTECO/CORTECO.bmp"
          }
          'msg' : "保存成功",
          'code' : 200 #状态码
        }
        image: 图片二进制bytes
         1001         图片保存失败
         1002         图片格式不对
         1003         服务器异常
         1004         图片内容／类型不能为空
        """
        if not all([filename, data, classfiy]):
            logging.info('调用图片接口缺少参数')
            raise BaseRPCError('缺少参数')
        rpc_url = IMAGE_HOST + self.uri
        files = ('data', filename, data)
        fields = (('itype', filename.split('.')[-1].encode('utf-8')), ('classify', classfiy))
        content_type, body = self.encode_multipart_formdata(fields, files)
        headers = {"Content-Type": content_type, 'content-length': str(len(body))}
        request = HTTPRequest(rpc_url, method="POST", headers=headers, body=body, validate_cert=False)
        response_json = await self.send_request(request)
        # 封装返回
        data = response_json.get('imginfo', {})
        code = response_json.get('code', 0)
        if code != 200:
            logging.info(response_json)
            raise BaseRPCError('图片服务异常')
        else:
            return data['remoteurl']

    @classmethod
    def encode_multipart_formdata(self, fields, files):
        # 封装multipart/form-data post请求
        boundary = b'WebKitFormBoundaryh4QYhLJ34d60s2tD'
        boundary_u = boundary.decode('utf-8')
        crlf = b'\r\n'
        l = []
        for (key, value) in fields:
            l.append(b'--' + boundary)
            temp = 'Content-Disposition: form-data; name="%s"' % key
            l.append(temp.encode('utf-8'))
            l.append(b'')
            if isinstance(value,str):
                l.append(value.encode())
            else:
                l.append(value)
        key, filename, value = files
        l.append(b'--' + boundary)
        temp = 'Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename)
        l.append(temp.encode('utf-8'))
        temp = 'Content-Type: img/%s' % filename.split('.')[1]
        l.append(temp.encode('utf-8'))
        l.append(b'')
        l.append(value)
        l.append(b'--' + boundary + b'--')
        l.append(b'')
        body = crlf.join(l)
        content_type = 'multipart/form-data; boundary=%s' % boundary_u
        return content_type, body
