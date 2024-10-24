import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from segmentation_models_pytorch.base import (
    SegmentationModel,
    SegmentationHead,
    ClassificationHead,
)

import random
from DnCNN import make_net
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.encoders.mobileone import reparameterize_model, MobileOneBlock
from deeplabv3 import DeepLabV3Decoder, DeepLabV3PlusDecoder, ASPP, SeparableConv2d
import numpy as np
import copy
from torchvision.models._utils import IntermediateLayerGetter


def get_pf_list():
    pf1 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 1, 0]]).astype('float32')

    pf2 = np.array([[0, 0, 0],
                    [0, -1, 1],
                    [0, 0, 0]]).astype('float32')

    pf3 = np.array([[0, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]]).astype('float32')

    return [torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone(),
            torch.tensor(pf1).clone(),
            torch.tensor(pf2).clone(),
            torch.tensor(pf3).clone()
            ]


class dlv3(nn.Module):
    def __init__(self,
                encoder_name="mobileone_s4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,                      # model output channels (number of classes in your dataset)
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                encoder_output_stride: int = 16,
                activation: Optional[str] = None,
                decoder_channels: int = 256,
                decoder_atrous_rates: tuple = (12, 24, 36),
                upsampling: int = 4,
                aux_params: Optional[dict] = None,
                inference_mode=False,
                replace=False, 
                index=0):
        super(dlv3, self).__init__()
        self.encoder_depth=encoder_depth

        self.normal_conv = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.pf_conv = nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=1, bias=False)
        
         
        num_levels = 17
        out_channel = 1
        self.dncnn = make_net(3, kernels=[3, ] * num_levels,
                        features=[64, ] * (num_levels - 1) + [out_channel],
                        bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                        acts=['relu', ] * (num_levels - 1) + ['linear', ],
                        dilats=[1, ] * num_levels,
                        bn_momentum=0.1, padding=1).cuda()
        dat = torch.load('./trufor.pth.tar', map_location=torch.device('cpu'))
        if 'network' in dat:
            dat = dat['network']
        dncnn_dict={}
        for i,j in dat['state_dict'].items():
            if 'dncnn' in i:
                dncnn_dict[i.replace('dncnn.', '')]=j
        self.dncnn.load_state_dict(dncnn_dict) 
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
            output_stride=encoder_output_stride
        )
        
        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )
        self.relu=nn.ReLU(inplace=True)
        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )
        # self.init_transformer()
        if aux_params is not None:
            self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        else:
            self.classification_head = None
            
        self.pf_list = get_pf_list()
        self.reset_pf()
    def reset_pf(self):
        for idx, pf in enumerate(self.pf_list):
            self.pf_conv.weight.data[idx, :, :, :] = pf
    def forward(self, x, noiseprinter_x=None, get_feature=True, continual=False, shot=False, t_mask=None, bs=None):
        
        if noiseprinter_x is None:
            noiseprinter_x = self.dncnn(x)
        normal_x=self.normal_conv(x)
        pf_x = self.pf_conv(x)
        noiseprinter_x = torch.tile(noiseprinter_x, (3, 1, 1))
        x = torch.cat([normal_x, pf_x, noiseprinter_x], dim=1)

        features = self.encoder(x)

        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            if get_feature:
                return masks, labels, features
            else:
                return masks, labels
        if get_feature:
            return masks, features
        else:
            return masks
    
    
    
    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x