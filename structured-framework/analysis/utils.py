import numpy as np

def tryconverttonp(num):
    try:
        return num.cpu().detach().numpy()
    except:
        #print(num)
        return num
