# -*- coding: utf-8 -*-
"""
    run this file to start server
    For Example
    $ python app.py --p 6666
"""
import logging
from tornado import httpserver, ioloop
from tornado.options import define, options

from core.application import BaseApplication
from apps import urls
import utils.logging

define("port", default=8888, type=int)
options.parse_command_line()

def make_app():
    server = httpserver.HTTPServer(BaseApplication(urls, port=options.port))
    return server

def main():
    server = make_app()
    server.listen(options.port, address="0.0.0.0")
    print("start success,the port is [%d]" % options.port)
    ioloop.IOLoop.instance().start()

if __name__ == '__main__':
    main()
