import os
import sys

sys.path.insert(1,os.getcwd())
import json
import random

# download train, test, and val split
def download_data():
    os.system('bash datasets/gqa_extras/gqadownload.sh')


class CLEVRDataset():
    # split = train/test/val
    def __init__(self,split='val'):
        #self.answermapping = []
        with open("data/raw/CLEVR_v1.0/questions/CLEVR_"+split+"_questions.json") as f:
            q = json.load(f)
        self.split = split

    def getdata(self,idx):
        question_info = self.q["questions"][idx]
        image_id = question_info['image_index']
        question = question_info['question']
        question_id = question_info['question_index']
        answer = question_info['answer']
        imgfile = "data/CLEVR_v1.0/images/"+split+"/CLEVR_"+split+f"_{image_id:06d}.png"
        return imgfile, question, answer

    def length(self):
        return len(self.q['questions'])

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




    
    


