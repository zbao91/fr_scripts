#!/bin/sh

echo '------start to align'
curl 'localhost:8888/img_processing/im_cropped'
echo 'image align finished'
echo '------start to calculate embedding'
curl 'localhost:8888/img_processing/cal_embd'
echo 'image embeddings calculation finished'
echo '------start to group face'
curl 'localhost:8888/img_processing/face_group'
echo 'image group by people finished'
echo '------start to delete single image'
curl 'localhost:8888/img_processing/delete_single'
echo 'delete single image finished'
