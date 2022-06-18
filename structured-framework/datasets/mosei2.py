import os
import sys

import random
import pickle
import torch
from models.mult.src.dataset import Multimodal_Datasets  

class MOSEIDataset():
    # only support split = val for now
    def __init__(self,split='valid'):
        self.split = split
        with open('data/MOSEI/mosei_valid_processed_features_list.pkl', 'rb') as f:
            self.dataset = pickle.load(f)
        self.listdata = list(self.dataset.items())   
        self.answermapping = ['positive', 'negative']
        self.fau_agg_dict = {
            "FaceEmotion": [0,1,2,3,4,5,6,7,8,9],
            "Brow": [10,11,12],
            "Eye": [13,14,15, 29],
            "Nose": [16],
            "Lip": [17,18,19,20,22,23,24,25,26,28],
            "Chin": [21,27],
            "HeadMovement":[32,33,34],
            "Others": [30,31]
        } 
        self.audio_agg_dict = {
            "pitch": [0],
            "glottal": [1,2,3,4,5,6,7,8,9,10],
            "amplitude": [11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35],
            "phase": [36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73]
        }
        self.FAU_LISTS= ['Anger', 'Contempt', 'Disgust', 'Joy', 'Fear', 'Baseline', 'Sadness', 'Surprise',
                         'Confusion', 'Frustration', 'AU1', 'AU2', 'AU4', 'AU5', 'AU6', 'AU7', 'AU9', 'AU10',
                         'AU12', 'AU14', 'AU15', 'AU17', 'AU18', 'AU20', 'AU23', 'AU24', 'AU25', 'AU26', 'AU28',
                         'AU43', 'Has_Glasses', 'Is_Male', 'Pitch', 'Yaw', 'Roll']
        self.AUDIO_LISTS = ["F0", "VUV", "NAQ", "QOQ", "H1H2", "PSP", "MDQ", "peakSlope", "Rd", "Rd_conf", "creak", "MCEP_0", "MCEP_1", "MCEP_2", "MCEP_3", "MCEP_4",
                            "MCEP_5", "MCEP_6", "MCEP_7", "MCEP_8", "MCEP_9", "MCEP_10", "MCEP_11", "MCEP_12", "MCEP_13", "MCEP_14", "MCEP_15", "MCEP_16", "MCEP_17",
                            "MCEP_18", "MCEP_19", "MCEP_20", "MCEP_21", "MCEP_22", "MCEP_23", "MCEP_24", "HMPDM_0", "HMPDM_1", "HMPDM_2", "HMPDM_3", "HMPDM_4",
                            "HMPDM_5", "HMPDM_6", "HMPDM_7", "HMPDM_8", "HMPDM_9", "HMPDM_10", "HMPDM_11", "HMPDM_12", "HMPDM_13", "HMPDM_14", "HMPDM_15", "HMPDM_16", 
                            "HMPDM_17", "HMPDM_18", "HMPDM_19", "HMPDM_20", "HMPDM_21", "HMPDM_22", "HMPDM_23", "HMPDM_24", "HMPDD_0", "HMPDD_1", "HMPDD_2", "HMPDD_3",
                            "HMPDD_4", "HMPDD_5", "HMPDD_6", "HMPDD_7", "HMPDD_8", "HMPDD_9", "HMPDD_10", "HMPDD_11", "HMPDD_12"]
        for k, v in self.fau_agg_dict.items():
            for i in v:
                self.FAU_LISTS[i] = k + ':' + self.FAU_LISTS[i] 
        for k, v in self.audio_agg_dict.items():
            for i in v:
                self.AUDIO_LISTS[i] = k + ':' + self.AUDIO_LISTS[i]         
        

    def getdata(self,idx):
        data = self.listdata[idx]
        name = data[0]
        embeds = data[1]['embeddings']
        text = torch.tensor(embeds['text']).cuda()
        audio = torch.tensor(embeds['audio']).cuda()
        vision = torch.tensor(embeds['vision']).cuda()
        label = data[1]['groundtruth'][0]
        words = data[1]['features']['words'][1]
        return text, audio, vision, label, name, words

    def getrawdata(self,idx):
        data = self.listdata[idx]
        return data   

    def length(self):
        return len(self.listdata)

    def classnames(self):
        return self.answermapping

    def sample(self,num):
        sampled=[]
        nums=[]
        for i in range(self.length()):
            nums.append(i)
        random.shuffle(nums)
        idx=0
        while(len(sampled) < num):
            a = self.getdata(nums[idx])
            sampled.append(a)
            idx += 1
        return sampled

    def get_feature_def(self):
        return self.FAU_LISTS, self.AUDIO_LISTS

    def get_correct_label(self,datainstance):
        if datainstance[3] >= 0:
            return 0
        return 1      

    def get_correct_answer(self, datainstance):
        label = self.get_correct_label(datainstance)
        return self.answermapping[label]    
  


class MOSEIDataset_orig():
    # split = train/test/valid
    def __init__(self,split='valid',if_align=True):
        self.split = split
        self.if_align = if_align
        if not if_align:
            self.datafile = pickle.load(open('data/MOSEI/mosei_senti_data_noalign.pkl', 'rb'))
        else:
            self.datafile = pickle.load(open('data/MOSEI/mosei_senti_data.pkl', 'rb'))    
        self.dataset = Multimodal_Datasets('data/MOSEI', data='mosei_senti', split_type=split, if_align=if_align)
         

    def getdata(self,idx):
        X, Y, META = self.dataset[idx]
        idx, text, audio, vision = X
        label = Y[0][0].item()
        name = META[0]
        return text, audio, vision, label, name, None

    def length(self):
        return len(self.dataset)

    def classnames(self):
        return None

    def sample(self,num):
        sampled=[]
        nums=[]
        for i in range(self.length()):
            nums.append(i)
        random.shuffle(nums)
        idx=0
        while(len(sampled) < num):
            a = self.getdata(nums[idx])
            sampled.append(a)
            idx += 1
        return sampled


    
    


