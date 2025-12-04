import numpy as np
import torch
import torch.nn as nn
import torch.nn. functional as F
import matplotlib
import matplotlib.pylab as plt
from PIL import Image
from torchvision import  transforms
import argparse

from cam import GradCAM, ScoreCAM
from methods import visualize

import pandas as pd

images_path = '../isic-data/ISIC2018_Task1-2_Training_Input/'
csv_path = '../isic-data/isic2018_T1T2.csv'
model_path = '../DL-Models-ISIC/results-comet-iv4/rn50_i18nf_5runs_5/checkpoints/model_best.pth'
results_path = 'rn50_res/'#'pix_attr_results/'

idx2label = {0:'Benign',1:'Melanoma'}

def load():
    df=pd.read_csv(csv_path)
    images =df.image.values
    images =[images[i][:-5] for i in range(len(images))]
    labels = df.label.values

    model = torch.load(model_path)
    model.eval()

    target_layer = model.layer4
    target_layer 
    return images, labels, model, target_layer 


def create_cam_explanation(index,model,target_layer,images, labels):
    """plot original image, heatmap from cam and superimpose image"""
    wrapped_model_sc = ScoreCAM(model, target_layer)
    wrapped_model_gc =GradCAM(model, target_layer)
    img_sample = Image.open(images_path+images[index]+'.jpg')
    cls_true = labels[index]
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    tensor = preprocess(img_sample)

    # reshape 4D tensor (N, C, H, W)
    tensor = tensor.unsqueeze(0)
    
    cam_gc, cls_pred = wrapped_model_gc(tensor.cuda())
    cam_sc, _ = wrapped_model_sc(tensor.cuda())
    heatmap_sc = visualize(tensor, cam_sc)
    heatmap_gc = visualize(tensor, cam_gc)
    
   
    imsv_sc = Image.fromarray((heatmap_sc*255).astype(np.uint8))
    imsv_gc = Image.fromarray((heatmap_gc*255).astype(np.uint8))

    if cls_true == 1 and cls_pred == 1:
        imsv_sc.save(results_path + '/True_Positive/'+images[index]+'_ScoreCAM.png')
        imsv_gc.save(results_path + '/True_Positive/'+images[index]+'_GradCAM.png')
    elif cls_true == 0 and cls_pred == 0:
        imsv_sc.save(results_path + '/True_Negative/'+images[index]+'_ScoreCAM.png')
        imsv_gc.save(results_path + '/True_Negative/'+images[index]+'_GradCAM.png')
    elif cls_true == 0 and cls_pred == 1:
        imsv_sc.save(results_path + '/False_Positive/'+images[index]+'_ScoreCAM.png')
        imsv_gc.save(results_path + '/False_Positive/'+images[index]+'_GradCAM.png')
    elif cls_true == 1 and cls_pred == 0:
        imsv_sc.save(results_path +  '/False_Negative/'+images[index]+'_ScoreCAM.png')
        imsv_gc.save(results_path +  '/False_Negative/'+images[index]+'_GradCAM.png')
    del cam_gc, cam_sc,wrapped_model_gc,wrapped_model_sc

def create_imgs(start_idx,end_idx= 0):
    images, labels, model, target_layer  = load()

    if end_idx== 0:
        end_idx= len(labels)
    print('Creating explanations for images:',start_idx,images[start_idx],'to',end_idx,images[end_idx-1])

    for i in range(start_idx,end_idx):
        create_cam_explanation(i,model,target_layer, images, labels)
        print('Explanation created for: ',i, images[i])   



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_idx', type=int, help='Image index to start')
    parser.add_argument('end_idx', type=int, help='Image index to stop')
    args = parser.parse_args()
    create_imgs(args.start_idx, args.end_idx)
    
