#!/bin/bash

mkdir -p data
mkdir -p data/flickr30k
cd data/flickr30k

git clone https://github.com/BryanPlummer/flickr30k_entities
unzip -n flickr30k_entities/annotations.zip
rm -rf flickr30k_entities
cd ../..