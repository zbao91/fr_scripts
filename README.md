# 华数人脸识别
说明: 目前采用开源的facenet和使用vggface2 pre-trained的模型
---
# 模块说明
## apps
功能模块，所有的接口都在此模块中。根据功能将不同的功能放在不同的目录中
* dir - face_recgnition: 人脸识别模块
    * class - FaceIdComparison: 人证对比
        * input: 2 images
        * output: dict
    * class - FaceValidate: 人脸验证 
        * 与人脸库（facebank）中的人脸进行校验
* .py - hander: 基础handler

## config
配置文件（暂时不用）

## core
web服务核心模块(暂时不用)

## facebank
人脸数据库, 文件夹格式如下:     
---facebank    
  |--- name1    
  |   |--- img1.jpg   
  |   |--- img2.jpg   
  |     
  |--- nam2     
  |   |--- img1.jpg   
  |   |--- img2.jpg   
  
 ## log
 日志文件
 * debug: debug日志
 * error: 错误日志
 * info: info日志
 
 ## middleware
 中间件，暂时不用
 
 ## models
 深度学习模型
 ### facenet
 Google facenet模型
 ### mtcnn
 mtcnn模型，用来进行人脸检测
    
## services
rpc微服务，暂时不需要

## utils
一些通用功能，暂时不需要