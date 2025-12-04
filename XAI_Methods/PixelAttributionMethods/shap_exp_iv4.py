import numpy as np
import torch

import torch.nn as nn
import torch.nn. functional as F
import matplotlib
import matplotlib.pylab as plt
from PIL import Image
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
import argparse
import pandas as pd
import shap

from plot_shap_dl import plot_single_exp

IMG_SIZE = (299,299)

images_path = '../isic-data/ISIC2018_Task1-2_Training_Input/'
csv_path = '../isic-data/isic2018_T1T2.csv'
model_path = '../DL-Models-ISIC/results-comet-iv4/iv4_i18nf_5runs_4/checkpoints/model_best.pth'

isic18_root="../isic-data/HAM10000/ISIC2018_Task3_Training_Input/"
train_i18_csv="../DL-Models-ISIC/csv_splits/isic2018n_train.csv"
results_path = 'pix_attr_results/'

df=pd.read_csv(csv_path)
images =df.image.values
labels = df.label.values

def load():
    df=pd.read_csv(csv_path)
    images =df.image.values
    images =[images[i][:-5] for i in range(len(images))]
    labels = df.label.values

    model = torch.load(model_path)
    model.eval()

    return images, labels, model

def preprocess_img(image_id,path):
    img_sample = Image.open(path+image_id+'.jpg')
    preprocess = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img = np.array(img_sample.resize(IMG_SIZE, Image.BILINEAR))
    #img = np.float32(img) / 255.0
    # convert image to tensor
    tensor = preprocess(img_sample)

    # reshape 4D tensor (N, C, H, W)
    #tensor = tensor.unsqueeze(0)
    return tensor

def get_shap_dl(model):

    df_train=pd.read_csv(train_i18_csv) # Data used to train the model

    images_train_comp=df_train.image_id.values
    labels_train_comp=df_train.label.values

    _, X_sample_train, _, _ = train_test_split(
    images_train_comp, labels_train_comp, test_size=132, random_state=42,shuffle= True, stratify= labels_train_comp)
    train_samp_imgs=[]
    for i in range(0,len(X_sample_train)):
        tmp = preprocess_img(X_sample_train[i],isic18_root)
        #print(tmp.shape)
        train_samp_imgs.append(tmp)
    train_samp_imgs = torch.stack(train_samp_imgs)
    S1 = shap.DeepExplainer(model,train_samp_imgs.to('cuda'))
    return S1


def create_shap_explanation(index,model,explainer,images, labels):
    tensor_img = preprocess_img(images[index],images_path)
    tensor_img = tensor_img.unsqueeze(0)
    score = model(tensor_img.cuda())

    prob = F.softmax(score, dim=1)

    prob, idx = torch.max(prob, dim=1)
    cls_pred = idx.item()
    cls_true = labels[index]
    prob = prob.item()
    print("predicted class ids {}\t probability {} True class: {}".format(cls_pred, prob,cls_true))
    
    

    ShapValues = explainer.shap_values(tensor_img.cuda())
    #print('ShapVals',np.array(ShapValues).shape)
    sv= np.moveaxis(ShapValues[cls_pred][0],0,-1)
    img_np= np.moveaxis(tensor_img.detach().numpy(),1,-1)
    #print(sv.shape)
    #print(img_np.shape)
    Plot = plot_single_exp(np.expand_dims(sv,axis=0),img_np,show =False)
    #Plot.savefig('lala1.png')

    if cls_true == 1 and cls_pred == 1:
        Plot.savefig(results_path + '/True_Positive/'+images[index]+'_SHAP.png')
       
    elif cls_true == 0 and cls_pred == 0:
        Plot.savefig(results_path + '/True_Negative/'+images[index]+'_SHAP.png')
        
    elif cls_true == 0 and cls_pred == 1:
        Plot.savefig(results_path + '/False_Positive/'+images[index]+'_SHAP.png')
        
    elif cls_true == 1 and cls_pred == 0:
        Plot.savefig(results_path +  '/False_Negative/'+images[index]+'_SHAP.png')




def create_imgs(start_idx):#,end_idx= 0):
    images, labels, model  = load()
    S1 = get_shap_dl(model)
    
    # if end_idx== 0:
    #     end_idx= len(labels)
    print('Creating explanations for images:',start_idx,images[start_idx])#,'to',end_idx,images[end_idx-1])
    create_shap_explanation(start_idx,model,S1, images, labels)
    print('Explanation created for: ',start_idx, images[start_idx]) 

    # for i in range(start_idx,end_idx):
    #     create_shap_explanation(i,model,S1, images, labels)
    #     print('Explanation created for: ',i, images[i])   



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_idx', type=int, help='Image index to start')
    #parser.add_argument('end_idx', type=int, help='Image index to stop')
    args = parser.parse_args()
    create_imgs(args.start_idx)#, args.end_idx)


# imgs_tensors_i18t1_t2 =[]
# for i in range(0,10):#len(images)):
#     tmp = preprocess_img(images[i],images_path)
#     #print(tmp.shape)
#     if i%100==0:
#         print(i)
#     imgs_tensors_i18t1_t2.append(tmp)
# imgs_tensors_i18t1_t2 = torch.stack(imgs_tensors_i18t1_t2)
# model = torch.load(model_path)
# model.eval()






# ShapValues = S1.shap_values(imgs_tensors_i18t1_t2.to('cuda'))
# print('ShapVals',np.array(ShapValues).shape)

##Plot by Shap Library

# shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in ShapValues]
# test_numpy = np.swapaxes(np.swapaxes(imgs_tensors_i18t1_t2.numpy(), 1, -1), 1, 2)

# shap.image_plot(shap_numpy, test_numpy,show = False)
# plt.savefig('scratch.png')

##----------------

#for i in images:


# class_idx = 0
# sample_idx = 0
# sv= np.moveaxis(ShapValues[class_idx][sample_idx],0,-1)
# img_np= np.moveaxis(imgs_tensors_i18t1_t2[0].detach().numpy(),0,-1)
# print(sv.shape)
# print(img_np.shape)
# Plot = plot_single_exp(np.expand_dims(sv,axis=0) ,np.expand_dims(img_np,axis=0),show =False)
# Plot.savefig('lala.png')

# score = model(tensor.cuda())

# prob = F.softmax(score, dim=1)

# prob, idx = torch.max(prob, dim=1)
# cls_pred = idx.item()
# prob = prob.item()
# print("mine-predicted class ids {}\t probability {}".format(cls_pred, prob))