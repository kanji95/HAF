import os

from glob import glob
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from util.trees import load_hierarchy
# from torchvision.datasets.folder import DatasetFolder, default_loader

class TieredImagenetH(Dataset):
    def __init__(self, root="/media/newhd/Imagenet2012/Imagenet-orig/", mode="train", transform=None, is_parent=False):
        
        imagenet_split = "train" if mode == "train" else "val"
        self.imagenet_dir = os.path.join(root, imagenet_split)
        
        self.is_parent = is_parent
        
        split_path = os.path.join("./data/splits_tieredimagenet/", mode)
        class_files = glob(split_path + "/*")
        
        self.classes = sorted([os.path.splitext(os.path.basename(cls_file))[0] for cls_file in class_files])
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        hierarchy = load_hierarchy("tiered-imagenet-224", "./data")
        n_leaves = len(hierarchy.leaves())
        leavepos = set(hierarchy.leaf_treeposition(n) for n in range(n_leaves))
        
        self.parent_classes = []
        self.parent_map = {}
        for i in range(len(leavepos)):
            self.parent_classes.append(hierarchy[list(leavepos)[i][:-1]].label())
            self.parent_map[hierarchy[list(leavepos)[i]]] = hierarchy[list(leavepos)[i][:-1]].label()
        self.parent_classes = sorted(list(set(self.parent_classes)))
        self.num_parent_classes = len(self.parent_classes)
        self.parent_class_to_idx = {cls_name: i for i, cls_name in enumerate(self.parent_classes)}
        
        self.reverse_mapping_index = defaultdict(list)
        for child_key in self.parent_map:
            self.reverse_mapping_index[self.parent_class_to_idx[self.parent_map[child_key]]].append(self.class_to_idx[child_key])
        
        if self.is_parent:

            self.classes = self.parent_classes
            self.num_classes = self.num_parent_classes
            self.class_to_idx = self.parent_class_to_idx
        
        self.samples = []
        for class_file in class_files:
            class_name = os.path.splitext(os.path.basename(class_file))[0]
            with open(class_file, 'r') as f:
                image_names = f.readlines()
            image_names = [image_name.strip() for image_name in image_names]
            
            if self.is_parent:
                class_samples = [(os.path.join(self.imagenet_dir, class_name, image_name), self.class_to_idx[self.parent_map[class_name]]) for image_name in image_names]
            else:
                class_samples = [(os.path.join(self.imagenet_dir, class_name, image_name), self.class_to_idx[class_name]) for image_name in image_names]
            self.samples.extend(class_samples)
        
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        path, classid = self.samples[index]
        sample = Image.open(path).convert("RGB")
        if self.transform:
            sample = self.transform(sample)
            
        # target = torch.zeros(self.num_classes)
        # target[classid] = 1 
    
        return sample, classid
