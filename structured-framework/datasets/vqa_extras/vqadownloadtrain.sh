#!/bin/bash

cd data

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip

unzip v2_Questions_Train_mscoco.zip

wget http://images.cocodataset.org/zips/train2014.zip

unzip train2014.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip

unzip v2_Annotations_Train_mscoco.zip

cd ..
