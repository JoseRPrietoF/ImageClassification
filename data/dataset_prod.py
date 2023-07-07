import pytorch_lightning as pl
import torch, glob, os
from torch.utils.data import DataLoader
import multiprocessing
# from models import Collator
import random, numpy as np
from datasets import load_dataset
from transformers import AutoFeatureExtractor
from torchvision.transforms import Compose, Normalize, RandomResizedCrop, ColorJitter, ToTensor
from torchvision import  transforms as transforms_tv
def worker_init_fn(_):
    # We need to reset the Numpy and Python PRNG, or we will get the
    # same numbers in each epoch (when the workers are re-generated)
    random.seed(torch.initial_seed() % 2 ** 31)
    np.random.seed(torch.initial_seed() % 2 ** 31)

class  ImageDatasetProd(pl.LightningDataModule):

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, opts=None, batch_size=64,  width=120, height=60, nchannels=1, work_dir="", img_dirs="", corpus="", args={}, model="resnet50", txt_path_tr:str="", txt_path_te:str="", nfeats:int=1024):
        super().__init__(train_transforms=train_transforms, val_transforms=val_transforms, test_transforms=test_transforms, dims=dims)
        # self.setup(opts)
        size=width
        self.opts = opts
        use_distortions = False
        prob_rand_aug = 0.5
        self.img_dirs = img_dirs
        self.width, self.height = width, height
        # self.img_dirs = "data/hisclima/feats"
        self.batch_size = batch_size
        self.work_dir = work_dir
        self.args = args
        self.corpus = corpus
        if "JMBD" in self.corpus:
            self.n_classes = 3
        elif "hisclima" in self.corpus:
            self.n_classes = 8
        if nchannels == 1:
            self.channels = "L"
        elif nchannels == 3:
            self.channels = "RGB"
        else:
            raise Exception
        # self.data_collator = DefaultDataCollator()
        if "resnet" in model:
            if "resnet50" == model:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-50", size=size)
            elif "resnet18" == model:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-18", size=size)
            elif "resnet101" == model:
                self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/resnet-101", size=size)
        elif "swintiny" in model:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224", size=size)
        elif "swinbase" in model:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224", size=size)
        elif "swinv2_base" in model:
            # self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window16-256", size=size)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft", size=size)
        elif "convnext" in model:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/convnext-tiny-224", size=size)
        if txt_path_tr:
            file_tfidf = self.process_text(txt_path_tr, txt_path_te, nfeats)
        else:
            file_tfidf = None
        self.collator = ImageClassificationCollator(self.feature_extractor, file_tfidf)

    
    def process_text(self, txt_tr, txt_te, nfeats:int):
        def read_tfidf_file(res:dict, p:str, nfeats:int):
            f = open(p, "r")
            lines = f.readlines()
            f.close()
            header = lines[0] # order of words
            lines = lines[1:]
            for line in lines:
                line = line.strip().split(" ")
                fname = "_".join(line[0].split("_")[:-1])
                feats = [float(x) for x in line[1:]][:nfeats]
                res[fname] = feats

        file_tfidf = {}
        read_tfidf_file(file_tfidf, txt_tr, nfeats)
        read_tfidf_file(file_tfidf, txt_te, nfeats)
        return file_tfidf

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
        print(self.dataset)
        self.label2tag = self.get_label_dict(self.dataset['test'])
        self.tag2label = {v:k for k,v in self.label2tag.items()}
        print(self.tag2label)

    def predict_dataloader(self):
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        te_dataset = self.dataset['test']
        # te_dataset.set_transform(self.transforms_te)
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
   def __init__(self, feature_extractor, file_tfidf:dict={}): 
      self.feature_extractor = feature_extractor
      self.file_tfidf = file_tfidf
   def __call__(self, batch):
        encodings = self.feature_extractor([x['image'] for x in batch],
        return_tensors='pt')   
        if self.file_tfidf is not None:
            encodings['text'] = torch.tensor([self.file_tfidf[x['image'].filename.split("/")[-1].split(".")[0]] for x in batch])
        return encodings

