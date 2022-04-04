import sys
import os
import numpy as np
from torch.utils.data import DataLoader
import random
import pickle
import copy


class MIMICDataset:
    def __init__(self,split='test',path='/home/paul/yiwei/im.pk'):
        if split == 'train':
            theindex = 0
        elif split == 'val':
            theindex=1
        elif split == 'test':
            theindex = 2
        else:
            raise NotImplementedError
        self.dataset = get_data(path)[theindex]
    def getdata(self,idx):
        return self.dataset[idx]
    def length(self):
        return len(self.dataset)
    def classnames(self):
        return ['not have disease','have disease']
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
    def allstatic(self):
        a=np.zeros((self.length(),5))
        for i in range(self.length()):
            a[i] = self.dataset[i][0]
        return a




#task: integer between -1 and 19 inclusive, -1 means mortality task, 0-19 means icd9 task
def get_data(imputed_path='im.pk',task=7):
  f = open(imputed_path,'rb')
  datafile = pickle.load(f)
  f.close()
  X_t = datafile['ep_tdata']
  X_s = datafile['adm_features_all']

  X_t[np.isinf(X_t)]=0
  X_t[np.isnan(X_t)]=0
  X_s[np.isinf(X_s)]=0
  X_s[np.isnan(X_s)]=0

  X_s_avg=np.average(X_s,axis=0)
  X_s_std=np.std(X_s,axis=0)
  X_t_avg=np.average(X_t,axis=(0,1))
  X_t_std=np.std(X_t,axis=(0,1))

  for i in range(len(X_s)):
    X_s[i] = (X_s[i]-X_s_avg)/X_s_std
    for j in range(len(X_t[0])):
      X_t[i][j] = (X_t[i][j]-X_t_avg)/X_t_std

  static_dim=len(X_s[0])
  timestep=len(X_t[0])
  series_dim=len(X_t[0][0])
  if task<0:
    y=datafile['adm_labels_all'][:,1]
    admlbl=datafile['adm_labels_all']
    le = len(y)
    for i in range(0,le):
      if admlbl[i][1]>0:
        y[i]=1
      elif admlbl[i][2]>0:
        y[i]=2
      elif admlbl[i][3]>0:
        y[i]=3
      elif admlbl[i][4]>0:
        y[i]=4
      elif admlbl[i][5]>0:
        y[i]=5
      else:
        y[i]=0
  else:
    y=datafile['y_icd9'][:,task]
    le = len(y)
  datasets=[[X_s[i],X_t[i],y[i]] for i in range(le)]
  random.seed(10)

  random.shuffle(datasets)
  return datasets[le//5:],datasets[0:le//10],datasets[le//10:le//5]
