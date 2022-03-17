from torchvision import transforms
from torch.utils.data.dataset import Dataset
import random
import os
import pickle
import json
import torch
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class CRFDataLoader(Dataset):
    def __init__(self, stage_to_number, input_size=224, 
                 train=True, set_name='training', num_samples=50):
        self.input_size = input_size
        self.train = train
        self.num_samples = num_samples
        self.stage_to_number = stage_to_number
        # Making transforms
        self.to_tensor = transforms.ToTensor()
        img_transformlist = []
        flow_transformlist = []
        transform_list = []
        transform_list += [transforms.Resize(self.input_size)]
        if train == True:
            transform_list = transform_list + [
                transforms.RandomResizedCrop(
                    self.input_size, scale=(0.8, 1.2)),
                transforms.RandomRotation((-20, 20),),  # 45, 45
                transforms.RandomHorizontalFlip(),
            ]
        flow_transformlist += transform_list
        img_transformlist += transform_list
        img_transformlist += [transforms.Normalize(mean=[0.5], std=[1])]

        self.img_transform = transforms.Compose(img_transformlist)
        self.flow_transform = transforms.Compose(flow_transformlist)


    def __getitem__(self, index):
        '''
        Write a custom dataloader
        img_tensors
            - input frames
            - dtype: torch.Tensor
            - size: [batch size, num sampled frames, 1, height, width]
        img_labels
            - per-frame ground truth stage labels
            - dtype: torch.Tensor
            - size: [batch size, num sampled frames]
        flow_tensors
            - consecutive two frames
            - dtype: torch.Tensor
            - size: [batch size, num sampled frames - 1, 2, height, width]
        flow_labels
            - transition labels
            - dtype: torch.Tensor
            - size: [batch_size, num sampled frames - 1]
        '''

        return img_tensors, img_labels, flow_tensors, flow_labels

    # def __len__(self):
    #     return 
