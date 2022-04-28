#!/bin/bash

mkdir -p data
mkdir -p data/flickr30k
cd data/flickr30k

git clone https://github.com/BryanPlummer/flickr30k_entities
unzip flickr30k_entities/annotations.zip
rm -r flickr30k_entities/annotations.zip

cd ../..