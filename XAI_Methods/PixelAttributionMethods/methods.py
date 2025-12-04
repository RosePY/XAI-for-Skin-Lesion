

import torch
import torch.nn.functional as F
import numpy as np
import cv2


def reverse_normalize(x1, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    x = x1.detach().clone()
    x[:, 0, :, :] = x[:, 0, :, :] * std[0] + mean[0]
    x[:, 1, :, :] = x[:, 1, :, :] * std[1] + mean[1]
    x[:, 2, :, :] = x[:, 2, :, :] * std[2] + mean[2]
    return x


def visualize(img, cam1):
    """
    Synthesize an image with CAM to make a result image.
    Args:
        img: (Tensor) shape => (1, 3, H, W)
        cam: (Tensor) shape => (1, 1, H', W')
    Return:
        synthesized image (Tensor): shape =>(1, 3, H, W)
    """

    _, _, H, W = img.shape
    img = reverse_normalize(img).cpu()
    cam = cam1.detach().clone()
    cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    cam = 255 * cam.squeeze()
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_VIRIDIS)
    heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    heatmap = heatmap.float() / 255
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])
    #print(heatmap)

    result = 0.85*heatmap+0.45*img.squeeze()
    result = result.div(result.max())


    # img = reverse_normalize(img)
    # img = cv2.cvtColor( img.squeeze().numpy().transpose(1, 2, 0),cv2.COLOR_BGR2RGB)
    # #cam = F.interpolate(cam, size=(H, W), mode='bilinear', align_corners=False)
    # cam = cv2.resize(cam, (W, H))
    # cam = 255 * cam.squeeze()
    # heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_VIRIDIS)
    
    # hif = .5
    # superimposed_img = heatmap * hif + img *0.5
    # superimposed_img = np.minimum(superimposed_img, 255.0).astype(np.uint8)  # scale 0 to 255  
    # superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    # # heatmap = torch.from_numpy(heatmap.transpose(2, 0, 1))
    # # heatmap = heatmap.float() / 255
    # # b, g, r = heatmap.split(1)
    # # heatmap = torch.cat([r, g, b])
    # # alpha = 
    # # result = ()heatmap + img.cpu()
    # # result = result.div(result.max())

    return result.squeeze().numpy().transpose(1, 2, 0)