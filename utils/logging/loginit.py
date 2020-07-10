#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018-12-26 22:12
# @Author   : cancan
# @File     : Log.py
# @Function : 

import os
import logging
DEBUG = True

from .loghandler import MultiProcessSafeTimeRotatingFileHandler


class LogInit(object):
    """日志模块

    目前默认按 1 天分割，保留 30 天日志，不同级别日志存储到不同目录，

    logging 模块中的 handlers 在线程中是安全的，在进程中是不安全的，如果需要同时写同一文件，
    需要重写 handler 或使用第三方包（例如：ConcurrentLogHandler）实现锁机制来解决写文件冲突。

    第三方包 ConcurrentLogHandler 不支持按时间分割，按时间分割需要需要自己重构，
    重构的多进程按时间分割模块在同目录下的 MPLogging.py 中
    """

    LOG_PATH = os.path.join(os.getcwd(), 'log')

    DEFAULT_FORMAT = \
        '[%(levelname)1.1s %(asctime)s %(filename)s:%(lineno)d] %(message)s'
    DEFAULT_DATE_FORMAT = '%y%m%d %H:%M:%S'

    # 支持的日志分离级别
    _support_level = [
        logging.getLevelName(logging.CRITICAL),
        logging.getLevelName(logging.INFO),
        logging.getLevelName(logging.ERROR),
        logging.getLevelName(logging.DEBUG),
    ]

    @classmethod
    def init(cls, filename, log_path=None, logger=None, clear_handlers=False,
             when='d', backup_count=30, **kwargs):

        """初始化配置

        初始化后，直接调用 Python 内置 logging 模块进行日志打印

        Args:
            filename: 日志文件名，不同级别日志会单独保存到对应级别文件夹
            logger: 自定义日志对象，默认处理根日志对象
            log_path: 日志文件存放的根目录，如果不传默认存放当前目录
            clear_handlers: 是否清楚原日志处理对象中的处理方法
            when: 时间分割规则，参数支持大小
                S: 按秒分割
                M: 按分钟分割
                D: 按天分割
            backup_count: 备份数量，超出备份数自动删除
            kwargs: 自定义级别分离存储，不设置则默认分离 info 和 error
                eg:
                    critical/CRITICAL  = True 则分离存储 critical 信息
                    info/INFO = True 则分离存储 info 信息
                    error/ERROR  = True 则分离存储 error 信息
                    debug/DEBUG = True 则分离存储 debug 信息
        """

        if log_path:
            cls.LOG_PATH = log_path

        if not logger:
            logger = logging.getLogger()

        # 是否清空原有的默认处理类
        if clear_handlers:
            logger.handlers = []

        peewee_logger = logging.getLogger('peewee')
        if DEBUG:
            peewee_logger.handlers = [cls._error_handler(filename, when, backup_count)]
        else:
            peewee_logger.handlers = [cls._debug_handler(filename, when, backup_count)]

        peewee_logger.setLevel(logging.DEBUG)

        # 没有指定需要分离存储的日志，则默认分离 info 和 error
        if not kwargs:
            logger.addHandler(cls._info_handler(filename, when, backup_count))
            logger.addHandler(cls._debug_handler(filename, when, backup_count))
            logger.addHandler(cls._error_handler(filename, when, backup_count))



        else:
            for level in kwargs:
                if level.upper() in cls._support_level:
                    if kwargs[level]:
                        logger.addHandler(
                            getattr(cls, '_%s_handler' % level.lower())(
                                filename, when, backup_count
                            )
                        )
                else:
                    raise ValueError('Not support log level : %s' % level)

    @classmethod
    def _format_log_filename(cls, level, filename):
        """格式化日志文件名"""

        filename_end = '_{filename}.log'.format(filename=filename)

        if DEBUG:
            filename = os.path.join(cls.LOG_PATH, 'error', 'error' + filename_end)
        else:
            if level == logging.CRITICAL:
                filename = \
                    os.path.join(cls.LOG_PATH, 'total', 'total' + filename_end)
            elif level == logging.INFO:
                filename = \
                    os.path.join(cls.LOG_PATH, 'info', 'info' + filename_end)
            elif level == logging.ERROR:
                filename = \
                    os.path.join(cls.LOG_PATH, 'error', 'error' + filename_end)
            elif level == logging.DEBUG:
                filename = \
                    os.path.join(cls.LOG_PATH, 'debug', 'debug' + filename_end)
            else:
                raise ValueError('Unsupported log level')
        cls._check_log_file_path(filename)
        return filename

    @classmethod
    def _check_log_file_path(cls, file_path):
        """检测日志文件目录和文件"""

        file_dir, filename = os.path.split(file_path)

        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        if not os.path.exists(file_path):
            f = open(file_path, 'w')
            f.close()

    @classmethod
    def _default_log_formatter(cls):
        """默认日志输出样式"""

        return logging.Formatter(
            fmt=cls.DEFAULT_FORMAT,
            datefmt=cls.DEFAULT_DATE_FORMAT
        )

    @classmethod
    def _total_handler(cls, server_name, when, backup_count):
        """总日志处理方法"""

        handler = MultiProcessSafeTimeRotatingFileHandler(
            filename=cls._format_log_filename(logging.CRITICAL, server_name),
            when=when,
            backupCount=backup_count
        )

        handler.setFormatter(cls._default_log_formatter())

        return handler

    @classmethod
    def _info_handler(cls, server_name, when, backup_count):
        """信息日志处理方法"""

        handler = MultiProcessSafeTimeRotatingFileHandler(
            filename=cls._format_log_filename(logging.INFO, server_name),
            when=when,
            backupCount=backup_count
        )

        handler.addFilter(cls._filter_log_level(logging.INFO))
        handler.setFormatter(cls._default_log_formatter())

        return handler

    @classmethod
    def _debug_handler(cls, server_name, when, backup_count):
        """调试日志处理方法"""

        handler = MultiProcessSafeTimeRotatingFileHandler(
            filename=cls._format_log_filename(logging.DEBUG, server_name),
            when=when,
            backupCount=backup_count
        )

        handler.addFilter(cls._filter_log_level(logging.DEBUG))
        handler.setFormatter(cls._default_log_formatter())

        return handler

    @classmethod
    def _error_handler(cls, server_name, when, backup_count):
        """错误日志处理方法"""

        handler = MultiProcessSafeTimeRotatingFileHandler(
            filename=cls._format_log_filename(logging.ERROR, server_name),
            when=when,
            backupCount=backup_count
        )

        handler.addFilter(cls._filter_log_level(logging.ERROR))
        handler.setFormatter(cls._default_log_formatter())

        return handler

    @classmethod
    def _filter_log_level(cls, level):
        """日志级别过滤"""

        handler_filter = logging.Filter()
        handler_filter.filter = lambda record: record.levelno == level

        return handler_filter
