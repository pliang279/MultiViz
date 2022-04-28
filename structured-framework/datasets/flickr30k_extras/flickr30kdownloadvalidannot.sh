#!/bin/bash

mkdir -p data
mkdir -p data/flickr30k
cd data/flickr30k

wget https://sid.erda.dk/share_redirect/CrLpUMgIKh/annotations/valid_ann.jsonl

cd ../..