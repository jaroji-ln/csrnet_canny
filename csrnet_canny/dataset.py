import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        # invoke load_data to get img and target
        img, target = load_data(img_path, self.train) 
    
        # Re-call images using cv for canny
        img_cv = cv2.imread(img_path)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    
        # Canny edge detection
        edge = cv2.Canny(gray, 100, 200)
        edge = edge.astype(np.float32) / 255.0  # Normalize images
        edge = np.expand_dims(edge, axis=0)  # (1, H, W)
        edge_tensor = torch.from_numpy(edge).float()
        
        if self.transform is not None:
            img_tensor = self.transform(img_cv)  # apply transform
        # concatenage RGB + Edge 
        img = torch.cat([img_tensor, edge_tensor], dim=0)  # (4, H, W)
        # img = img_tensor
        return img, target