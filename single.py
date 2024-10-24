import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
from self_adapt_bn2 import replace_batchnorm, SelfAdaptiveNormalization
import copy
import numpy as np
from dlv3 import dlv3
import imageio
device = torch.device('cuda')
def rgba2rgb(self, rgba, background=(255, 255, 255)):
    row, col, ch = rgba.shape

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

    a = np.asarray(a, dtype='float32') / 255.0

    R, G, B = background

    rgb[:, :, 0] = r * a + (1.0 - a) * R
    rgb[:, :, 1] = g * a + (1.0 - a) * G
    rgb[:, :, 2] = b * a + (1.0 - a) * B

    return np.asarray(rgb, dtype='uint8')
def test(image_name):

    # define backbone
    FENet_name = 'mobileone'
    FENet = dlv3(
        encoder_name="mobileone_s4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=15,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,                      # model output channels (number of classes in your dataset)
        activation='sigmoid',
        inference_mode=True,
    )

    FENet = FENet.to(device)
    FENet_weight_path = './FT_CMT_mobileone.pth'
    
    state_dict = torch.load(FENet_weight_path, map_location='cuda')
    
    new_state={}
    for i,j in state_dict.items():
        if 'all_running_mean' in i or 'all_running_var' in i or 'classification_head' in i:
            continue
        else:
            new_state[i]=j
    state_dict=new_state

    cur_domain_id=10
    replace_batchnorm(FENet.encoder, cur_domain_id)
    replace_batchnorm(FENet.decoder, cur_domain_id)
    replace_batchnorm(FENet.segmentation_head, cur_domain_id)
    
    FENet.load_state_dict(state_dict,strict=True)


    VOCdevkit_path="/home/tmp/idcl/dataset/"
    image = imageio.imread(image_name)
    if image.shape[-1] == 4:
        image = self.rgba2rgb(image)
    image = torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)
    image = image.cuda().unsqueeze(0)
    with torch.no_grad():
        FENet.eval()
        pred_mask, _ = FENet(image)

        pred_mask = F.interpolate(pred_mask, size=(image.size(2), image.size(3)), mode='bilinear',
                                align_corners=True)
    print(pred_mask)
    return pred_mask

    
if __name__ == '__main__':
    test('/home/tmp/coda/PSCC-Net/results/image_results/COCO_train2014_000000087056.png')