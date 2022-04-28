#!/bin/bash

mkdir -p data
mkdir -p data/flickr30k
cd data/flickr30k

folder="flickr30k_entities"
if ! git clone "${https://github.com/BryanPlummer/flickr30k_entities}" "${folder}" 2>/dev/null && [ -d "${folder}" ] ; then
    echo "Clone failed because the folder ${folder} exists"
fi
unzip -n flickr30k_entities/annotations.zip
rm -rf flickr30k_entities
cd ../..