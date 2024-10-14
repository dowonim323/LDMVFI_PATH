import numpy as np
import random
from os import listdir
from os.path import join, isdir, split, getsize
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from PIL import Image
import ldm.data.vfitransforms as vt
from functools import partial
import torch
from torchvision.transforms import ToPILImage

class otls_triplet(Dataset):
    def __init__(self, db_dir, train=True,  crop_sz=(256,256), augment_s=True, augment_t=True, step = 20):
        
        self.crop_sz = crop_sz
        self.augment_s = augment_s
        self.augment_t = augment_t
        self.step = step

        self.data = torch.load(join(db_dir, 'false_color.pt'))

    def __getitem__(self, index):
        
        z_size, x_size, y_size = self.data.shape[:-1]
        
        x_index_max = (2 * (x_size - 1 - self.crop_sz[0])) // self.crop_sz[0]
        y_index_max = (2 * (y_size - 1 - self.crop_sz[1])) // self.crop_sz[1]
        
        y_index = index % (y_index_max + 1)
        index = index // (y_index_max + 1)
        
        x_index = index % (x_index_max + 1)
        index = index // (x_index_max + 1)
        
        z_index = index
                
        rawFrame1 = ToPILImage()(self.data[z_index, (x_index*self.crop_sz[0])//2:(x_index*self.crop_sz[0])//2+self.crop_sz[0], (y_index*self.crop_sz[1])//2:(y_index*self.crop_sz[1])//2+self.crop_sz[1], :].permute(2, 0, 1))
        rawFrame2 = ToPILImage()(self.data[z_index+self.step, (x_index*self.crop_sz[0])//2:(x_index*self.crop_sz[0])//2+self.crop_sz[0], (y_index*self.crop_sz[1])//2:(y_index*self.crop_sz[1])//2+self.crop_sz[1], :].permute(2, 0, 1))
        rawFrame3 = ToPILImage()(self.data[z_index+2*self.step, (x_index*self.crop_sz[0])//2:(x_index*self.crop_sz[0])//2+self.crop_sz[0], (y_index*self.crop_sz[1])//2:(y_index*self.crop_sz[1])//2+self.crop_sz[1], :].permute(2, 0, 1))

        if self.augment_s:
            rawFrame1, rawFrame2, rawFrame3 = vt.rand_flip(rawFrame1, rawFrame2, rawFrame3, p=0.5)
        
        if self.augment_t:
            rawFrame1, rawFrame2, rawFrame3 = vt.rand_reverse(rawFrame1, rawFrame2, rawFrame3, p=0.5)

        to_array = partial(np.array, dtype=np.float32)
        frame1, frame2, frame3 = map(to_array, (rawFrame1, rawFrame2, rawFrame3)) #(256,256,3), 0-255

        frame1 = frame1/127.5 - 1.0
        frame2 = frame2/127.5 - 1.0
        frame2 = frame3/127.5 - 1.0

        return {'image': frame1, 'prev_frame': frame2, 'next_frame': frame3}

    def __len__(self):
        return ((self.data.shape[0] - 2* self.step) * ((2 * (self.data.shape[1] - 1 - self.crop_sz[0])) // self.crop_sz[0] + 1) * ((2 * (self.data.shape[2] - 1 - self.crop_sz[1])) // self.crop_sz[1] + 1))
    
class otls_train_triplet(Dataset):
    def __init__(self, db_dir, crop_sz=[256,256], p_datasets=None, iter=False, samples_per_epoch=1000):
        otls_train = otls_triplet(db_dir, train=True,  crop_sz=crop_sz)

        self.datasets = [otls_train]
        self.len_datasets = np.array([len(dataset) for dataset in self.datasets])
        self.p_datasets = p_datasets
        self.iter = iter

        if p_datasets is None:
            self.p_datasets = self.len_datasets / np.sum(self.len_datasets)

        self.samples_per_epoch = samples_per_epoch

        self.accum = [0,]
        for i, length in enumerate(self.len_datasets):
            self.accum.append(self.accum[-1] + self.len_datasets[i])

    def __getitem__(self, index):
        if self.iter:
            # iterate through all datasets
            for i in range(len(self.accum)):
                if index < self.accum[i]:
                    return self.datasets[i-1].__getitem__(index-self.accum[i-1])
        else:
            # first sample a dataset
            dataset = random.choices(self.datasets, self.p_datasets)[0]
            # sample a sequence from the dataset
            return dataset.__getitem__(random.randint(0,len(dataset)-1))
            

    def __len__(self):
        if self.iter:
            return int(np.sum(self.len_datasets))
        else:
            return self.samples_per_epoch