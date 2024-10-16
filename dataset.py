import cv2
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

ia.seed(1)    
seq = iaa.Sequential([iaa.Sharpen((0.0, 1.0)),  iaa.ElasticTransformation(alpha=50, sigma=5), iaa.Affine(rotate=(-45, 45)),
                      iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Crop(percent=(0, 0.1))], random_order=True)
    
################################  semi train dataset  ###########################
class semi_BaseDataSets(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'skin' or self.dataset == 'idrid':
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.dataset == 'polyp' or self.dataset == 'drive':
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        label = cv2.imread(self.base_dir + '/masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)
        
        prob = random.random()
        if prob>0.5:
            segmap = SegmentationMapsOnImage(np.array(label), shape=image.shape)
            image, label = seq(image=image, segmentation_maps=segmap)
            label = label.get_arr()
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        if self.dataset == 'idrid':
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        else:
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        sample = {'image': image, 'label': label, "idx": name}
        return sample

class TwoStreamBatchSampler(Sampler):
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self): 
        primary_iter = iterate_once(self.primary_indices) 
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (primary_batch + secondary_batch for (primary_batch, secondary_batch) 
                in zip(grouper(primary_iter, self.primary_batch_size),grouper(secondary_iter, self.secondary_batch_size)))

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable) 

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)

################################  test dataset  ###########################
class testBaseDataSets(Dataset):
    def __init__(self, base_dir, list_name,image_size,dataset):
        self.base_dir = base_dir
        self.sample_list = []
        self.h, self.w = image_size
        self.dataset = dataset
        with open(self.base_dir + list_name, 'r') as f1:
            self.sample_list = f1.readlines()
        self.sample_list = [item.replace('\n', '')  for item in self.sample_list]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        if self.dataset == 'skin' or self.dataset == 'idrid':
            image = cv2.imread(self.base_dir + '/images/'+name+'.jpg', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.dataset == 'polyp' or self.dataset == 'drive':
            image = cv2.imread(self.base_dir + '/images/'+name+'.png', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        label = cv2.imread(self.base_dir + '/masks/'+name+'.png', cv2.IMREAD_GRAYSCALE)
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        if self.dataset == 'idrid':
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        else:
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        
        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        sample = {'image': image, 'label': label, "idx": name}
        return sample

################################  fully supvised  ######################################       

class MyDataSet(Dataset):
    def __init__(self, root, list_name, train_num, image_size,dataset):
        self.root = root
        self.list_path = self.root + list_name
        self.h, self.w = image_size
        self.dataset = dataset
        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        if train_num>700:
            self.img_ids = self.img_ids
        else:
            self.img_ids = self.img_ids[:train_num]
        self.files = []
        for name in self.img_ids:
            if self.dataset == 'skin' or self.dataset == 'idrid': 
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            elif self.dataset == 'polyp' or self.dataset == 'drive':
                img_file = os.path.join(self.root, "images/%s.png" % name)
            else:
                img_file = os.path.join(self.root, "images/%s.jpg" % name)
            label_file = os.path.join(self.root, "masks/%s.png" % name)
            self.files.append({"img": img_file,"label": label_file, "name": name})
        #np.random.shuffle(self.files)
        print("total {} samples".format(len(self.files)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        ###  aug 
        
        prob = random.random()
        if prob>0.5:
            segmap = SegmentationMapsOnImage(np.array(label), shape=image.shape)
            image, label = seq(image=image, segmentation_maps=segmap)
            label = label.get_arr()
        
        image = cv2.resize(image,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        if self.dataset == 'idrid':
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)
        else:
            label = cv2.resize(label,(self.w, self.h),interpolation = cv2.INTER_NEAREST)/255
        name = datafiles["name"]

        image = np.asarray(image, np.float32)
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image.astype(np.float32))
        label = torch.from_numpy(label.astype(np.uint8)).long()
        return image, label
