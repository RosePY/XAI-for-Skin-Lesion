import os
import sys
import random
import itertools
import colorsys

import numpy as np
from PIL import Image
import pandas as pd


images_folder = '../isic-data/ISIC2018_Task1-2_Training_Input/'
csv_path = '../isic-data/isic2018_T1T2.csv'
masks_folder = '../isic-data/ISIC2018_Task2_Training_GroundTruth_v3/'

def load():
    df=pd.read_csv(csv_path)
    images =df.image.values
    images =[images[i][:-5] for i in range(len(images))]
    return images


#melanoma_images_names = ['ISIC_0000301', 'ISIC_0000307', 'ISIC_0014979', 'ISIC_0000294', 'ISIC_0014525', 'ISIC_0000298', 'ISIC_0000043', 'ISIC_0000469', 'ISIC_0000004', 'ISIC_0000035', 'ISIC_0000306', 'ISIC_0014982', 'ISIC_0014527', 'ISIC_0014513', 'ISIC_0000310', 'ISIC_0000056', 'ISIC_0000487', 'ISIC_0011329', 'ISIC_0000550', 'ISIC_0000078', 'ISIC_0000077', 'ISIC_0000299', 'ISIC_0014506', 'ISIC_0000022', 'ISIC_0014951', 'ISIC_0000484', 'ISIC_0000303', 'ISIC_0000466', 'ISIC_0000143', 'ISIC_0000482', 'ISIC_0014543', 'ISIC_0000139', 'ISIC_0014897', 'ISIC_0000040', 'ISIC_0000290', 'ISIC_0014928', 'ISIC_0011315', 'ISIC_0014542', 'ISIC_0014507', 'ISIC_0014931', 'ISIC_0014985', 'ISIC_0000142', 'ISIC_0014987', 'ISIC_0000046', 'ISIC_0000313', 'ISIC_0014872', 'ISIC_0014548', 'ISIC_0000300', 'ISIC_0014912', 'ISIC_0000030']
#melanoma_images_names = ['ISIC_0012633']

images_patches = ['ISIC_0001163', 'ISIC_0001181', 'ISIC_0001184', 'ISIC_0001185', 'ISIC_0001186', 'ISIC_0001188', 'ISIC_0001190', 'ISIC_0001191', 'ISIC_0001204', 'ISIC_0001212', 'ISIC_0001213', 'ISIC_0001216', 'ISIC_0001242', 'ISIC_0001247', 'ISIC_0001254', 'ISIC_0001262', 'ISIC_0001267', 'ISIC_0001275', 'ISIC_0001286', 'ISIC_0001292', 'ISIC_0001296', 'ISIC_0001299', 'ISIC_0001306', 'ISIC_0001367', 'ISIC_0001372', 'ISIC_0001374', 'ISIC_0001385', 'ISIC_0001423', 'ISIC_0001427', 'ISIC_0001442', 'ISIC_0001449', 'ISIC_0001769', 'ISIC_0001852', 'ISIC_0001871', 'ISIC_0001960', 'ISIC_0002093', 'ISIC_0002107', 'ISIC_0002206', 'ISIC_0002246', 'ISIC_0002251', 'ISIC_0002287', 'ISIC_0002353', 'ISIC_0002374', 'ISIC_0002438', 'ISIC_0002439', 'ISIC_0002453', 'ISIC_0002459', 'ISIC_0002469', 'ISIC_0002476', 'ISIC_0002488', 'ISIC_0002489', 'ISIC_0002616', 'ISIC_0002647', 'ISIC_0002673', 'ISIC_0002780', 'ISIC_0002806', 'ISIC_0002829', 'ISIC_0002836', 'ISIC_0002871', 'ISIC_0002879', 'ISIC_0002885', 'ISIC_0002948', 'ISIC_0002975', 'ISIC_0002976', 'ISIC_0003005', 'ISIC_0003056', 'ISIC_0003174', 'ISIC_0003308', 'ISIC_0003346', 'ISIC_0003462', 'ISIC_0003539', 'ISIC_0003559', 'ISIC_0003582', 'ISIC_0003657', 'ISIC_0003728', 'ISIC_0003805', 'ISIC_0004110', 'ISIC_0004115', 'ISIC_0004168', 'ISIC_0004309', 'ISIC_0004337', 'ISIC_0004346', 'ISIC_0004715', 'ISIC_0004985', 'ISIC_0005000', 'ISIC_0005187', 'ISIC_0005247', 'ISIC_0005548', 'ISIC_0005555', 'ISIC_0005620', 'ISIC_0005639', 'ISIC_0005666', 'ISIC_0005787', 'ISIC_0006021', 'ISIC_0006114', 'ISIC_0006193', 'ISIC_0006350', 'ISIC_0006612', 'ISIC_0006651', 'ISIC_0006671', 'ISIC_0006711', 'ISIC_0006776', 'ISIC_0006795', 'ISIC_0006800', 'ISIC_0006815', 'ISIC_0006914', 'ISIC_0006940', 'ISIC_0006982', 'ISIC_0007038', 'ISIC_0007087', 'ISIC_0007141', 'ISIC_0007156', 'ISIC_0007241', 'ISIC_0007322', 'ISIC_0007332', 'ISIC_0007344', 'ISIC_0007475', 'ISIC_0007528', 'ISIC_0007557', 'ISIC_0007693', 'ISIC_0007760', 'ISIC_0007788', 'ISIC_0007796', 'ISIC_0008029', 'ISIC_0008116', 'ISIC_0008145', 'ISIC_0008207', 'ISIC_0008236', 'ISIC_0008256', 'ISIC_0008280', 'ISIC_0008294', 'ISIC_0008347', 'ISIC_0008396', 'ISIC_0008403', 'ISIC_0008406', 'ISIC_0008507', 'ISIC_0008524', 'ISIC_0008528', 'ISIC_0008541', 'ISIC_0008552', 'ISIC_0008600', 'ISIC_0008626', 'ISIC_0008659', 'ISIC_0008785', 'ISIC_0008807', 'ISIC_0008879', 'ISIC_0008913', 'ISIC_0008992', 'ISIC_0008993', 'ISIC_0008998', 'ISIC_0009035', 'ISIC_0009078', 'ISIC_0009083', 'ISIC_0009160', 'ISIC_0009165', 'ISIC_0009188', 'ISIC_0009201', 'ISIC_0009252', 'ISIC_0009297', 'ISIC_0009298', 'ISIC_0009344', 'ISIC_0009430', 'ISIC_0009504', 'ISIC_0009533', 'ISIC_0009564', 'ISIC_0009583', 'ISIC_0009599', 'ISIC_0009758', 'ISIC_0009800', 'ISIC_0009860']

attributes = {
    "globules": (255,90,255), # lila rosado fuerte
    "milia_like_cyst": (51,51,255), # azul morado
    "negative_network": (0,255,255), #celeste brilloso
    "pigment_network": (76,153,0), # verde claro brilloso
    "streaks": (255,0,0) # rojo
}


def apply_mask(image, mask, color, alpha=0.45):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask==255, image[:, :, c] * (1 - alpha) + alpha * color[c], image[:, :, c])
    return image

def read_image(image_path,num_chan, show = 0):
    with open(image_path, 'rb') as f:
        np_image_string = np.array([f.read()])    
    image = Image.open(image_path)
    width, height = image.size
    #print(width,height)
    
    
    if num_chan == 1:
        np_image = np.array(image.getdata()).reshape(height, width).astype(np.uint8)
    if num_chan == 3:
        np_image = np.array(image.getdata()).reshape(height, width, num_chan).astype(np.uint8)

    if show:
        display.display(display.Image(image_path, width=1024))
    return np_image

def save_image(new_image,output_image_path, show):   
    Image.fromarray(new_image.astype(np.uint8)).save(output_image_path)
    if show:
        display.display(display.Image(output_image_path, width=1024))






melanoma_images_names = load()
out_image_folder = '../isic-data/ISIC2018_T2_masked_imgs/'
cont = 0
for image_name in images_patches:#melanoma_images_names:

    image_path = images_folder+image_name+'.jpg'
    image = read_image(image_path,3)
    for key in attributes:
        mask_path = masks_folder+image_name+'_attribute_'+key+'.png'
        mask_image = read_image(mask_path, 1)
        image = apply_mask(image, mask_image, attributes[key])
    output_image_path = out_image_folder + image_name + '_attributes.jpg'
    save_image(image,output_image_path,0)
    if cont%10==0:
        print(cont)
    cont +=1


# image_folder = 'SOURCE_DIR/melanoma/'
# masks_folder =  'SOURCE_DIR/melanoma_attributes/'
# original_images = []
# original_images_attributes = []

# for image_name in melanoma_images_names:
#     image_path = image_folder+image_name+'.jpg'
#     image = read_image(image_path,3)
#     original_images.append(image)

# for image_name in melanoma_images_names:
#     image_path = masks_folder+image_name+'_atrributes.jpg'
#     image = read_image(image_path,3)
#     original_images_attributes.append(image)
