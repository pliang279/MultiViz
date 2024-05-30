
import os
import sys
import os
import sys

sys.path.insert(1, os.getcwd())

from datasets.vqa import VQADataset
from models.vqa_lxmert import VQALXMERT
sys.path.insert(1, os.getcwd())
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
from models.lxmert_extras import utils
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
import pickle as pkl
import pandas as pd

def store_pkl(split, model, datas):
    q = None
    a = None
    if split == "val":
        with open("data/v2_OpenEnded_mscoco_" + split + "2014_questions.json") as file1:
            q = json.load(file1)
        with open("data/v2_mscoco_" + split + "2014_annotations.json") as file2:
            a = json.load(file2)
        df = pd.DataFrame(columns=['question_id', 'image_id', 'image_path', 'question', 'gt_answer_str', 'gt_answer_class', 'lxmert_answer_str', 'lxmert_answer_class', 'lxmert_features'])
        f = 1
        length = len(q["questions"])
        for i in range(0, length):
            instance_info = datas.getdata(i, True)
            instance = datas.getdata(i)
            out = model.forward(instance)
            pred_idx = analysismodel.getpredlabel(out)
            pred = datas.classnames()[pred_idx]
            new_row = {'question_id': instance_info[0], 'image_id':instance_info[1], 'image_path':instance_info[2], 'question':instance_info[3], 'gt_answer_str':instance_info[4], 'gt_answer_class':instance_info[5], 'lxmert_answer_str':pred, 'lxmert_answer_class':pred_idx, 'lxmert_features':out[1].cpu()}
            df = df.append(new_row, ignore_index=True)
            if i == (length// 2) or i == length-1:
                df.to_pickle('tmp/VQAfeats/LXMERT_'+split+'_feats_'+f+'.pkl')
                df = pd.DataFrame(columns=['question_id', 'image_id', 'image_path', 'question', 'gt_answer_str', 'gt_answer_class', 'lxmert_answer_str', 'lxmert_answer_class', 'lxmert_features'])
                f+=1
    elif split == "train":
        with open("data/v2_OpenEnded_mscoco_" + split + "2014_questions.json") as file1:
            q = json.load(file1)
        with open("data/v2_mscoco_" + split + "2014_annotations.json") as file2:
            a = json.load(file2)
        df = pd.DataFrame(columns=['question_id', 'image_id', 'image_path', 'question', 'gt_answer_str', 'gt_answer_class', 'lxmert_answer_str', 'lxmert_answer_class', 'lxmert_features'])
        f = 1
        length = len(q["questions"])
        for i in range(0, length):
            instance_info = datas.getdata(i, True)
            instance = datas.getdata(i)
            out = model.forward(instance)
            pred_idx = analysismodel.getpredlabel(out)
            pred = datas.classnames()[pred_idx]
            new_row = {'question_id': instance_info[0], 'image_id':instance_info[1], 'image_path':instance_info[2], 'question':instance_info[3], 'gt_answer_str':instance_info[4], 'gt_answer_class':instance_info[5], 'lxmert_answer_str':pred, 'lxmert_answer_class':pred_idx, 'lxmert_features':out[1].cpu()}
            df = df.append(new_row, ignore_index=True)
            if i == (length// 3) or i == (length)//3 *2 or i == length-1:
                df.to_pickle('tmp/VQAfeats/LXMERT_'+split+'_feats_'+f+'.pkl')
                df = pd.DataFrame(columns=['question_id', 'image_id', 'image_path', 'question', 'gt_answer_str', 'gt_answer_class', 'lxmert_answer_str', 'lxmert_answer_class', 'lxmert_features'])
                f+=1

if __name__ == '__main__':
    split = "train"
    d=VQADataset(split)
    analysismodel = VQALXMERT()
    store_pkl(split, analysismodel, d)
    print("done")

