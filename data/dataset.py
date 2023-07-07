import pytorch_lightning as pl
import torch
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

class  ImageDataset(pl.LightningDataModule):

    def __init__(self, train_transforms=None, val_transforms=None, test_transforms=None, dims=None, opts=None, batch_size=64,  width=120, height=60, nchannels=1, work_dir="", img_dirs="", corpus="", args={}, model="resnet50", txt_path_tr:str="", txt_path_te:str="", nfeats:int=1024, n_classes=3):
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
        self.n_classes = n_classes
        # elif "hisclima" in self.corpus:
        #     self.n_classes = 8
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

        # normalize = Normalize(mean=self.feature_extractor.image_mean, std=self.feature_extractor.image_std)
        # self._transforms_te = Compose(
        #     [
        #         ToTensor(), 
        #         normalize,
        #     ]
        # )
        # self._transforms = Compose(
        #     [
        #         # transforms_tv.RandomApply(
        #         #     transforms=[
        #         #         # RandomResizedCrop(self.feature_extractor.size, scale=(0.08, 1.0)), 
        #         #         # ColorJitter(brightness=0.5, hue=0.5), 
        #         #         # transforms_tv.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        #         #         # transforms_tv.RandomAutocontrast()
        #         #     ],
        #         #     p=prob_rand_aug
        #         # ),
        #         ToTensor(), 
        #         normalize,
        #     ]
        # )
        
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
        print(f"----------------------------------------------- {self.img_dirs}")
        self.dataset = load_dataset("imagefolder", data_dir=self.img_dirs)

        self.label2tag = self.get_label_dict(self.dataset['validation'])
        self.tag2label = {v:k for k,v in self.label2tag.items()}

    def train_dataloader(self):
        # trainloader_train = torch.utils.data.DataLoader(self.tr_dataset, batch_size=self.opts.batch_size, shuffle=True, num_workers=0)
        num_workers = multiprocessing.cpu_count()
        # num_workers = 1
        tr_dataset = self.dataset['train']
        # tr_dataset.set_transform(self.transforms)
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
        # val_dataset.set_transform(self.transforms_te)
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
        # te_dataset.set_transform(self.transforms_te)
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
    
    # def transforms(self,examples):
    #     # examples["image_orig"] = [image for image in examples["image"]]
    #     examples["image"] = [self._transforms(image.convert("RGB")) for image in examples["image"]]
    #     # for image in examples["image"]:
    #     #     image.load(self._transforms(image.convert("RGB")))
    #     return examples
    
    # def transforms_te(self,examples):
    #     # for image in examples["image"]:
    #     #     print(image)
    #     #     image.paste(image)
    #     #     print(image)
    #     # exit()
    #     # examples["image_orig"] = [image for image in examples["image"]]
    #     examples["image"] = [self._transforms_te(image.convert("RGB")) for image in examples["image"]]
    #     # for image in examples["image"]:
    #     #     image.load(self._transforms_te(image.convert("RGB")))
    #     return examples

class ImageClassificationCollator:
   def __init__(self, feature_extractor, file_tfidf:dict={}): 
      self.feature_extractor = feature_extractor
      self.file_tfidf = file_tfidf
   def __call__(self, batch):
        encodings = self.feature_extractor([x['image'] for x in batch],
        return_tensors='pt')   
        encodings['labels'] = torch.tensor([x['label'] for x in batch],    
        dtype=torch.long)
        if self.file_tfidf is not None:
            # arr = []
            # for x in batch:
            #     fname = x['image'].filename.split("/")[-1].split(".")[0]
            #     arr.append(self.file_tfidf[fname])
            # print(batch[0]['image'].filename.split("/")[-1].split(".")[0])
            # print(self.file_tfidf[batch[0]['image'].filename.split("/")[-1].split(".")[0]])
            encodings['text'] = torch.tensor([self.file_tfidf[x['image'].filename.split("/")[-1].split(".")[0]] for x in batch])
        return encodings

