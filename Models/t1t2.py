

import numpy as np
import skimage.transform
import torch
import torch.nn as nn
import torch.nn. functional as F
import matplotlib
import matplotlib.pylab as plt
from PIL import Image
from matplotlib.pyplot import imshow
from torchvision import models, transforms
from torchvision.utils import save_image
import pandas as pd


images_path = '../isic-data/ISIC2018_Task1-2_Training_Input/'
csv_path = '../isic-data/isic2018_T1T2.csv'
model_path_rn50 = 'results-comet-iv4/rn50_i18nf_5runs_5/checkpoints/model_best.pth'
model_path_iv4 = '../DL-Models-ISIC/results-comet-iv4/iv4_i18nf_5runs_4/checkpoints/model_best.pth'

results_path = 'results_pred/'#'pix_attr_results/'


df=pd.read_csv(csv_path)
images =df.image.values
images =[images[i][:-5] for i in range(len(images))]
labels = df.label.values


model_iv4 = torch.load(model_path_iv4)
model_iv4.eval()

model_rn50 = torch.load(model_path_rn50)
model_rn50.eval()


IMG_SIZE = (224,224)

idx2label = {0:'Benign',1:'Melanoma'}


def create_imgs(index):
    """plot original image, heatmap from cam and superimpose image"""
    

    img_sample = Image.open(images_path+images[index]+'.jpg')
    cls_true = labels[index]

    preprocess = transforms.Compose([
        transforms.Resize((299,299)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = np.array(img_sample.resize(IMG_SIZE, Image.BILINEAR))
    #img = np.float32(img) / 255.0
    # convert image to tensor
    tensor = preprocess(img_sample)

    
    # reshape 4D tensor (N, C, H, W)
    tensor = tensor.unsqueeze(0)

    #iv4
    score_i = model_iv4(tensor.cuda())

    prob_i = F.softmax(score_i, dim=1)

    prob_i, idx_i = torch.max(prob_i, dim=1)
    cls_pred_i = idx_i.item()
    prob_i = prob_i.item()
    #print("iv4-predicted class ids {}\t probability {}".format(cls_pred_i, prob_i))


    #rn50

    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = preprocess(img_sample)

    
    # reshape 4D tensor (N, C, H, W)
    tensor = tensor.unsqueeze(0)

    score_r = model_rn50(tensor.cuda())

    prob_r = F.softmax(score_r, dim=1)

    prob_r, idx_r = torch.max(prob_r, dim=1)
    cls_pred_r = idx_r.item()
    prob_r = prob_r.item()
    #print("rn50-predicted class ids {}\t probability {}".format(cls_pred_r, prob_r))


    return cls_true,cls_pred_i,prob_i,cls_pred_r,prob_r


results=[]
for i in range(len(images)):
    a,b,c,d,e = create_imgs(i)
    results.append([images[i],a,b,c,d,e])
    if i%100==0:
        print(i)

df = pd.DataFrame(results, columns=['image', 'Label_true','Label_pred_iv4','Prob_pred_iv4','Label_pred_rn50','Prob_pred_rn50'])
df.to_csv('./isic2018_T1T2_results_complete.csv',index=False)
