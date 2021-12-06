import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import *
import cv2
import os
import shutil
import numpy as np


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
     '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
     '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
     '新',
     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
     'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
     'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
     'V', 'W', 'X', 'Y', 'Z', 'I',  '-'
     ]


dict = {'A01':'京','A02':'津','A03':'沪','B02':'蒙',
        'S01':'皖','S02':'闽','S03':'粤','S04':'甘',
        'S05': '贵', 'S06': '鄂', 'S07': '冀', 'S08': '黑', 'S09': '湘',
        'S10': '豫', 'S12': '吉', 'S13': '苏', 'S14': '赣', 'S15': '辽',
        'S17': '川', 'S18': '鲁', 'S22': '浙',
        'S30':'渝', 'S31':'晋', 'S32':'桂', 'S33':'琼', 'S34':'云', 'S35':'藏',
        'S36':'陕','S37':'青', 'S38':'宁', 'S39':'新'}

dict_chars = {chars: i for i , chars in enumerate(CHARS)}


def label_encode(input_label):
    
    out_label = np.zeros([len(input_label)])
    for i, text in enumerate(input_label):
        out_label[i] = dict_chars[text]
        
    return out_label


def decode_label(out_tensor):
    
    pass
    
        