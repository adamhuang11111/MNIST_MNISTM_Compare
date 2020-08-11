import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.insert(0, '/Users/yangzhihan/code/python_modules/')
from path_magic import path_magic

from torch_utils.cnn import fmap_shape_from_tuple, fmap_shape_from_layer, fmap_shape_from_sequential

import os
path_magic.make_dir_importable_relative(current_fname=os.path.realpath(__file__), 
                                        nup=1,
                                        target_dir_name='scripts',
                                        show=False)

##### feature extractor class #####

class FeatureExtractor(nn.Module):

    def __init__(self, input_shape=(3, 28, 28)):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
        )
        fmap_shape_from_sequential(self.main, input_shape)
        
    def forward(self, images):
        features = self.main(images)
        bs = features.size(0)
        return features.view(bs, -1)
    
##### label classifier class #####
    
class LabelClassifier(nn.Module):

    def __init__(self, input_shape=576, nclass=10):
        super().__init__()
        self.nclass = nclass
        self.main = nn.Sequential(
            nn.Linear(input_shape, 64, bias=True),
            nn.ReLU(),
            
            nn.Linear(64, nclass, bias=True),
        )
        fmap_shape_from_sequential(self.main, input_shape)
        
    def forward(self, features):
        logits = self.main(features)
        bs = logits.size(0)
        return logits.view(bs, self.nclass)
    
##### discriminator class #####
    
class Dis(nn.Module):

    def __init__(self, input_shape=576, nclass=2):
        super().__init__()
        self.nclass = nclass
        self.main = nn.Sequential(
            nn.Linear(input_shape, 64, bias=True),
            nn.ReLU(),
            
            nn.Linear(64, nclass, bias=True),
        )
        fmap_shape_from_sequential(self.main, input_shape)
        
    def forward(self, features):
        logits = self.main(features)
        bs = logits.size(0)
        return logits.view(bs, self.nclass)