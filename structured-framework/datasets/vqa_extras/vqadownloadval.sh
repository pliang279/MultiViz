#!/bin/bash

cd data

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip

unzip v2_Questions_Val_mscoco.zip

wget http://images.cocodataset.org/zips/val2014.zip

unzip val2014.zip

wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

unzip v2_Annotations_Val_mscoco.zip

cd ..
