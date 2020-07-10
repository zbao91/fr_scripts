# -*- coding: utf-8 -*-
"""
-------------------------------------------------
    File Name:   errors
    Description: 异常
    Author:      wzj
    Date:        2019/7/23
-------------------------------------------------
    Change Activity:
-------------------------------------------------
"""
import logging

from .rcode import NO_PERMISSION, NO_LOGIN

class BaseError(Exception):

    code = 0
    msg = '服务异常'

    def __init__(self, msg=None, **kwargs):
        if msg:
            setattr(self, 'msg', msg)
        else:
            msg = self.msg
        super().__init__(msg)
        if 'code' in kwargs:
            setattr(self, 'code', kwargs['code'])


# --- 参数，app有关 ----
class BaseAPPError(BaseError):
    pass


class MissArgError(BaseAPPError):
    code = 4
    def __init__(self, argument_name):
        self.msg = '缺少参数' + argument_name + '或参数异常'
        super().__init__()


class BadArgError(BaseAPPError):
    code = 4


# --- 用户模块 ----
class BaseUserError(BaseError):
    pass


# --- 远程调用异常 ---

class BaseRPCError(BaseError):
    pass


class UserRoleError(BaseError):
    code = -999
    msg = '暂无权限'

class LoginNeedException(BaseError):
    msg = '登录异常'
    code = NO_LOGIN

class LoginTokenException(BaseError):
    msg = '登录异常'
    code = NO_LOGIN

class RolePermmisonExecption(BaseError):
    msg = '暂无权限'
    code = NO_PERMISSION

class ValidError(BaseError):
    code = 0
    msg = '输入异常'

