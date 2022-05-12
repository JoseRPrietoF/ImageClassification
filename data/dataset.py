import pytorch_lightning as pl
from sklearn.utils import shuffle
import data.transforms as transforms
import torch
from torch.utils.data import DataLoader
import multiprocessing
# from models import Collator
import random, numpy as np
import glob, os
from torch.utils.data import Dataset
from datasets import load_dataset
import cv2
from PIL import Image
import torchvision
from transformers import DefaultDataCollator
from transformers import AutoFeatureExtractor

def worker_init_fn(_):
    # We need to reset the Numpy and Python PRNG, or we will get the
    # same numbers in each epoch (when the workers are re-generated)
    random.seed(torch.initial_seed() % 2 ** 31)
    np.random.seed(torch.initial_seed() % 2 ** 31)

class  ImageDataset(pl.LightningDataModule):

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, opts=None, batch_size=64,  width=120, height=60, nchannels=1, work_dir="", img_dirs="", corpus="", args={}):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        # self.setup(opts)
        size=512
        self.opts = opts
        use_distortions = False
        self.img_dirs = img_dirs
        self.width, self.height = width, height
        # self.img_dirs = "data/hisclima/feats"
        self.batch_size = batch_size
        self.work_dir = work_dir
        self.args = args
        self.corpus = corpus
        if self.corpus == "JMBD4949":
            self.n_classes = 3
        if nchannels == 1:
            self.channels = "L"
        elif nchannels == 3:
            self.channels = "RGB"
        else:
            raise Exception
        # self.data_collator = DefaultDataCollator()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50", size=size)
        self.collator = ImageClassificationCollator(self.feature_extractor)


    def get_label_dict(self, d):
        """
        I'm sure there is any other way to do it...
        """
        res = {}
        for i in d:
            fname = i['image'].filename
            label = i['label']
            fname_l = fname.split("/")[-2]
            res[fname_l] = label
        return res

    def setup(self, stage):
        print("-----------------------------------------------")
        self.dataset = load_dataset("imagefolder", data_dir=self.img_dirs)
        self.label2tag = self.get_label_dict(self.dataset['validation'])
        self.tag2label = {v:k for k,v in self.label2tag.items()}
        print(self.dataset)
        
    def train_dataloader(self):
        # trainloader_train = torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.opts.batch_size, shuffle=True, num_workers=0)
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        tr_dataset = self.dataset['train']
        # tr_dataset.with_transform(transforms_f)
        tr_dataset_loader = DataLoader(
            dataset= tr_dataset,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collator
        )
        return tr_dataset_loader
    
    def val_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        # val_dataset = tDataset(self.val_data, transform=self.default_img_transform, height = self.height, width = self.width)
        val_dataset = self.dataset['validation']
        # val_dataset.with_transform(transforms_f)
        
        # return val_dataset
        val_dataset_loader = DataLoader(
            dataset=val_dataset ,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collator
        )
        return val_dataset_loader
    
    def test_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        te_dataset = self.dataset['test']
        te_dataset_loader = DataLoader(
            dataset=te_dataset ,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collator
        )
        return te_dataset_loader
    
    def predict_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        te_dataset = self.dataset['test']
        p_dataset_loader = DataLoader(
            dataset=te_dataset ,
            batch_size=self.batch_size,
            num_workers=num_workers,
            shuffle=False,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collator
        )
        return p_dataset_loader

class ImageClassificationCollator:
   def __init__(self, feature_extractor): 
      self.feature_extractor = feature_extractor
   def __call__(self, batch):  
        encodings = self.feature_extractor([x['image'] for x in batch],
        return_tensors='pt')   
        encodings['labels'] = torch.tensor([x['label'] for x in batch],    
        dtype=torch.long)
        #encodings['filenames'] = [x['image'].filename for x in batch]
        # for x in batch:
        #     print(x['image'].filename)
        return encodings