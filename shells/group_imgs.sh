#!/bin/sh

curl 'localhost:8888/im_processing/im_cropped'
echo 'image align finished'
curl 'localhost:8888/im_processing/cal_embd'
echo 'image embeddings calculation finished'
curl 'localhost:8888/im_processing/face_group'
echo 'image group by people finished'
