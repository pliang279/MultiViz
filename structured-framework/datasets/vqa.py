import os
import sys

sys.path.insert(1,os.getcwd())
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"
from models.lxmert_extras import utils
import json
import random
from PIL import Image
import matplotlib.pyplot as plt
# download train split or val split
def download_data(split = 'val'):
    os.system('bash datasets/vqa_extras/vqadownload'+split+'.sh')


class VQADataset():
    def __init__(self,split='val'):
        self.answermapping = utils.get_data(VQA_URL)
        with open('data/v2_OpenEnded_mscoco_'+split+'2014_questions.json') as f:
            self.q = json.load(f)
        with open('data/v2_mscoco_'+split+'2014_annotations.json') as f:
            self.a = json.load(f)
        self.split=split
    def getdata(self,idx):
        qinfo = self.q['questions'][idx]
        image_id = qinfo['image_id']
        ques = qinfo['question']
        qid = qinfo['question_id']
        ainfo = self.a['annotations'][idx]
        aword = ainfo['multiple_choice_answer']
        assert qid == ainfo['question_id']
        imgfile = f'data/{self.split}2014/COCO_{self.split}2014_{image_id:012d}.jpg'
        try:
            label = self.answermapping.index(aword)
        except:
            print("Warning: no LXMERT label for this point")
            return imgfile,ques,aword,None
        return imgfile,ques,aword,label
    def length(self):
        return len(self.q['questions'])
    def classnames(self):
        return self.answermapping
    def sample(self,num,noNone=True):
        sampled=[]
        nums=[]
        for i in range(self.length()):
            nums.append(i)
        random.shuffle(nums)
        idx=0
        while(len(sampled) < num):
            a = self.getdata(nums[idx])
            if a[-1] is not None or not noNone:
                sampled.append(a)
            idx += 1
        return sampled
    def getseqdata(self,start,end):
        a=[]
        for i in range(start,end):
            a.append(self.getdata(i))
        return a
    def makepic(self,id,pr=None):
        plt.clf()
        d=self.getdata(id)
        imgfile,ques,aword,label = d
        plt.imshow(Image.open(imgfile))
        title="Question: "+ques+"\nid: "+str(id)+" Correct answer: "+aword
        if pr is not None:
            title+=(" Pred Answer: "+self.classnames()[pr])
        plt.title(title)
        plt.savefig("visuals/data/vqa-"+self.split+"-"+str(id)+".png")



    
    


