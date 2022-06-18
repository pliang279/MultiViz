import matplotlib.pyplot as plt
from PIL import Image
import torch
import sys
import os
import subprocess
import cv2
from imutils import face_utils
import numpy as np
import argparse
import dlib

from datasets.mosei2 import*
from models.mosei_mult import*
from visualizations.visualizemosei import*

vision_mapping = {
    "FaceEmotion": ['jaw'],
    "Brow": ['left_eyebrow', 'right_eyebrow'],
    "Eye": ['left_eye', 'right_eye'],
    "Nose": ['nose'],
    "Lip": ['mouth'],
    "Chin": ['jaw'],
    "HeadMovement":['jaw'],
    "Has_Glasses": ['glasses'],
    "Is_Male": ['gender']
}

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/Raw/shape_predictor_68_face_landmarks.dat")

# unzip val videos
def unzip_videos(dataset):
    for i in range(dataset.length()):
        data = dataset.getrawdata(i)
        vname = data[0] + '.mp4'
        if not os.path.exists('data/Raw/Videos/' + vname):
            subprocess.check_call('unzip -j data/CMU_MOSEI.zip Raw/Videos/Full/Combined/' + vname + ' -d data/Raw/Videos', shell=True)

# unzip val transcript
def unzip_transcripts(dataset):
    for i in range(dataset.length()):
        data = dataset.getrawdata(i)
        fname = data[0] + '.txt'
        if not os.path.exists('data/Raw/Transcripts/' + fname):
            subprocess.check_call('unzip -j data/CMU_MOSEI.zip Raw/Transcript/Segmented/Combined/' + fname + ' -d data/Raw/Transcripts', shell=True)            


# draw top-k features to the video
def process_data(dataset, model, idx, topk=3, feat=None, backward=None, target_idxs=None, reverse=False):
    data = dataset.getrawdata(idx)
    datainstance = dataset.getdata(idx)
    info = data[1]

    # Calculate and group feature gradients
    if feat != None:
        text_grad, _ = model.getgrad(datainstance, 'text', feat, True)
        #audio_grad = model.getgrad(datainstance, 'audio', feat, True)
        vision_grad, _ = model.getgrad(datainstance, 'vision', feat, True)
    elif target_idxs != None:    
        text_grad, _ = model.getgrad(datainstance, 'text')
        #audio_grad = model.getgrad(datainstance, 'audio')
        vision_grad = model.getdoublegrad(datainstance, 'vision', target_idxs)
    else:
        if not reverse:
            text_grad, _ = model.getgrad(datainstance, 'text')
            vision_grad, _ = model.getgrad(datainstance, 'vision')
        else:
            text_grad, _ = model.getgrad(datainstance, 'text', reverse=True)
            vision_grad, _ = model.getgrad(datainstance, 'vision', reverse=True)
    text_grad_norm = torch.norm(text_grad[0], p=1, dim=1)
    Z_vision = torch.absolute(vision_grad[0].T)
    #Z_audio = torch.absolute(audio_grad[0].T)
    Z_vision_normed = torch.div(Z_vision, text_grad_norm).cpu().numpy()
    #Z_audio_normed = torch.div(Z_audio, text_grad_norm).cpu().numpy()

    vname = data[0] + '.mp4'
    outvname_1 = 'mosei_grad_'+str(idx) if backward == None else 'mosei_grad_'+str(backward)
    if reverse:
        outvname_1 = 'mosei_correct_grad_'+str(idx) if backward == None else 'mosei_correct_grad_'+str(backward)
    outvname_2 = '' if feat == None else '_feat_'+str(feat)
    outvname_3 = '' if backward == None else '_sample_'+str(idx)
    outvname = outvname_1 + outvname_2 + outvname_3 + '.mp4'
    out_dir_name = 'private_test_scripts/mosei_simexp/mosei'+str(idx)+'/'
    if backward != None:
        out_dir_name = 'private_test_scripts/mosei_simexp/mosei'+str(backward)+'/'

    vidcap = cv2.VideoCapture('data/Raw/Videos/' + vname)
    if (vidcap.isOpened()== False):
        print("Error opening video stream or file")
    fps = vidcap.get(cv2.CAP_PROP_FPS)    
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    print(out_dir_name + outvname)
    out = cv2.VideoWriter(out_dir_name + 'tmp_' + outvname, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))    

    start = info['start']
    end = info['end']
    segments = [start + i*(end-start)/50 for i in range(51)]
    intervals = [(segments[i], segments[i+1]) for i in range(len(segments)-1)]
    
    success,image = vidcap.read()
    count = 0
    success = True
    curr = 0
    frame_num = 0
    while success:
        success,frame = vidcap.read()
        if not success:
            break
        count+=1
        timestamp = count/fps
        interval = intervals[curr]

        if timestamp < interval[0]:
            continue

        if timestamp >= interval[1] and curr < len(intervals):
            curr += 1
            if curr >= len(intervals):
                break

        # Annotate the current frame
        if timestamp >= interval[0] and timestamp < interval[1]:
            # Compute feature group with max gradient
            sum_dict = dict()
            for k, v in fau_agg_dict_2.items():
                sum_dict[k] = np.sum(Z_vision_normed[:, curr][v])
            sum_list = list(sum_dict.items())
            topk_pairs = sorted(sum_list, key=lambda x: x[1])[len(sum_list)-topk:len(sum_list)]
            topk_names = [x[0] for x in topk_pairs]
            organ_names = [vision_mapping[name] for name in topk_names]

            # detect facial landmarks
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)

            # loop over the face detections
            for (_, rect) in enumerate(rects):
                # determine the facial landmarks for the face region, then
                # convert the landmark (x, y)-coordinates to a NumPy array
                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                # loop over the face parts individually
                for m in range(len(organ_names)):
                    organ = organ_names[m]
                    feat_group = topk_names[m]
                    # FAU Category = 'Others'
                    if organ[0] == 'glasses':
                        cv2.putText(frame, 'glasses', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        continue
                    elif organ[0] == 'gender':
                        cv2.putText(frame, 'gender', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        continue        
                    shape_idxs = []
                    for name in organ:
                        i, j = face_utils.FACIAL_LANDMARKS_IDXS[name]
                        # extract the ROI of the face region as a separate image
                        shape_idxs += range(i, j)

                    (x, y, w, h) = cv2.boundingRect(np.array([shape[shape_idxs]]))
                    cv2.rectangle(frame, (x, y, w, h), (36,255,12))
                    if feat_group == 'Chin':
                        cv2.putText(frame, feat_group, (x, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    elif feat_group == 'HeadMovement':    
                        cv2.putText(frame, feat_group, (x, y+h+17), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    else:
                        cv2.putText(frame, feat_group, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    
            cv2.imwrite('private_test_scripts/mosei_simexp/'+str(curr)+'.png', frame)    
            frame_num += 1
            out.write(frame)           

    vidcap.release()
    out.release()
    cv2.destroyAllWindows()
    os.system("ffmpeg -y -i " + out_dir_name + 'tmp_' + outvname + " -vcodec libx264 " + out_dir_name + outvname)
    os.system("rm " + out_dir_name + 'tmp_' + outvname)

# clip the video. Only keep the interval covered in the data.
def clip_data(dataset, model, idx, backward=None):
    data = dataset.getrawdata(idx)
    #datainstance = dataset.getdata(idx)
    info = data[1]

    vname = data[0] + '.mp4'
    out_dir_name = 'private_test_scripts/mosei_simexp/mosei'+str(idx)+'/'
    if backward != None:
        out_dir_name = 'private_test_scripts/mosei_simexp/mosei'+str(backward)+'/'

    vidcap = cv2.VideoCapture('data/Raw/Videos/' + vname)
    if (vidcap.isOpened()== False):
        print("Error opening video stream or file")
    fps = vidcap.get(cv2.CAP_PROP_FPS)    
    frame_width = int(vidcap.get(3))
    frame_height = int(vidcap.get(4))
    out = cv2.VideoWriter(out_dir_name + 'tmp_' + vname, cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width,frame_height))   
    print(out_dir_name + vname) 

    start = info['start']
    end = info['end']
    segments = [start + i*(end-start)/50 for i in range(51)]
    intervals = [(segments[i], segments[i+1]) for i in range(len(segments)-1)]
    
    success,image = vidcap.read()
    count = 0
    success = True
    curr = 0
    while success:
        success,frame = vidcap.read()
        if not success:
            break
        count+=1
        timestamp = count/fps
        interval = intervals[curr]

        if timestamp < interval[0]:
            continue

        if timestamp >= interval[1] and curr < len(intervals):
            curr += 1
            if curr >= len(intervals):
                break

        # keep this frame
        if timestamp >= interval[0] and timestamp < interval[1]:
            out.write(frame)           

    vidcap.release()
    out.release()
    cv2.destroyAllWindows()
    os.system("ffmpeg -y -i " + out_dir_name + 'tmp_' + vname + " -vcodec libx264 " + out_dir_name + vname)
    os.system("rm " + out_dir_name + 'tmp_' + vname)


# get raw script
def get_script(dataset, model, idx):
    data = dataset.getrawdata(idx)
    fname = data[0] + '.txt'
    info = data[1]
    with open('data/Raw/Transcripts/' + fname) as f:
        lines = f.readlines()
    script = None    
    for line in lines:
        parts = line.split('___')
        start = float(parts[2])
        if start == info['start']:
            script = parts[4]
            break
    return script        
     
