#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 2018-12-27 17:14
# @Author   : cancan
# @File     : MPlLogging.py
# @Function : 


import re
import time
import codecs
import os
from logging import FileHandler

COMPILE_FLAG = 256


class MultiProcessSafeTimeRotatingFileHandler(FileHandler):
    def __init__(self, filename, when='h', backupCount=5,
                 encoding=None, delay=0):
        """
        Use the specified filename for streamed logging
        """
        if codecs is None:
            encoding = None

        self.encoding = encoding
        self.suffix_time = ''
        self.baseFileDir = os.path.split(filename)[0]

        self.backupCount = backupCount
        self.when = when.upper()

        if self.when == 'S':
            self.suffix = "%Y-%m-%d_%H-%M-%S"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'M':
            self.suffix = "%Y-%m-%d_%H-%M"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}(\.\w+)?$"
        elif self.when == 'H':
            self.suffix = "%Y-%m-%d_%H"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}_\d{2}(\.\w+)?$"
        elif self.when == 'D' or self.when == 'MIDNIGHT':
            self.suffix = "%Y-%m-%d"
            self.extMatch = r"^\d{4}-\d{2}-\d{2}(\.\w+)?$"
        else:
            raise ValueError(
                "Invalid rollover interval specified: %s" % self.when)

        self.baseFilenameEndLen = len('.' + self.suffix) + 2

        self.extMatch = re.compile(self.extMatch, COMPILE_FLAG)

        FileHandler.__init__(self, filename, 'a', encoding, delay)

        # 清除基础文件名的文件因为实际不写入该文件中
        self.clearBaseFile()

    def clearBaseFile(self):
        """清除基础本文件"""

        try:
            os.remove(self.baseFilename)
        except FileNotFoundError:
            pass

    def emit(self, record):
        """
        Emit a record.

        Always check time
        """
        try:
            if self.check_baseFilename(record):
                self.build_baseFilename()
            FileHandler.emit(self, record)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)

    def check_baseFilename(self, record):
        """
        Determine if builder should occur.

        record is not used, as we are just comparing times,
        but it is needed so the method signatures are the same
        """
        timeTuple = time.localtime()

        if self.suffix_time != time.strftime(self.suffix, timeTuple) or \
                not os.path.exists(self.baseFilename + '.' + self.suffix_time):
            return 1
        else:
            return 0

    def build_baseFilename(self):
        """
        do builder; in this case,
        old time stamp is removed from filename and
        a new time stamp is append to the filename
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        # remove old suffix
        if self.suffix_time != "":
            self.baseFilename = self.baseFilename[:-self.baseFilenameEndLen]

        self.suffix_time = time.strftime(self.suffix, time.localtime())
        self.baseFilename = self.baseFilename + "." + self.suffix_time

        if not self.delay:
            self.stream = self._open()

        if self.backupCount > 0:
            for s in self.getFilesToDelete():
                try:
                    os.remove(s)
                except FileNotFoundError:
                    pass

    def getFilesToDelete(self):
        """
        Determine the files to delete when rolling over.

        More specific than the earlier method, which just used glob.glob().
        """
        dirName, baseName = os.path.split(self.baseFilename)
        fileNames = os.listdir(dirName)
        result = []
        # prefix = baseName + "."
        prefix = baseName[:-self.baseFilenameEndLen] + '.'
        plen = len(prefix)
        for fileName in fileNames:
            if fileName[:plen] == prefix:
                suffix = fileName[plen:]
                if self.extMatch.match(suffix):
                    result.append(os.path.join(dirName, fileName))
        if len(result) < self.backupCount:
            result = []
        else:
            result.sort()
            result = result[:len(result) - self.backupCount]
        return result[:-1]

    def _open(self):
        """
        Open the current base file with the (original) mode and encoding.
        Return the resulting stream.
        """

        if not os.path.exists(self.baseFileDir):
            os.makedirs(self.baseFileDir)

        return codecs.open(self.baseFilename, self.mode, encoding=self.encoding)
