
import data.transforms as transforms
import torch
import multiprocessing
import os
import random
import numpy as np
# from models.ctc_loss import CTCLoss
# from models.ctc_greedy_decoder import CTCGreedyDecoder
import torch
import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from data import ImageDataset
from models import Net
from torch import nn, save
from utils.functions import save_to_file
import shap


steps=[10]
seed = 1
# work_JMBD4949_4950_256feats_1example_rnnunits_64_rnnlayers3_cnnFalse_RGB/checkpoints/exp1_exp1/0_1oguiho2/checkpoints/epoch=18512-step=18512.ckpt
# checkpoint_load = "work_JMBD4949_4950_1024feats_rnnunits_256_rnnlayers3_cnnFalse_RGB_relative_split_notForced_noBI/checkpoints/exp1_exp1/0_edjf8iut/checkpoints/epoch=499-step=63999.ckpt"
checkpoint_load = "/data2/jose/projects/image_classif/work_JMBD4949_4950_tr49_resnet50/checkpoints/exp_JMBD4949_4950_tr49_resnet50_size1024_exp_JMBD4949_4950_tr49_resnet50_size1024/3_2qdjex9p/checkpoints/epoch=11-step=3299.ckpt"
do_train = True
model = "resnet50"

# corpus = f"hisclima"
# img_dirs = f"/home/jose/projects/image_classif/data/Hisclima"

tr_="tr49"
corpus = f"JMBD4949_4950_{tr_}"
img_dirs = f"/home/jose/projects/image_classif/data/JMBD4949_4950/{tr_}"

# corpus = f"JMBD4949"
# img_dirs = "/home/jose/projects/image_classif/data/{}".format(corpus)

gpu = 1
batch_size = 2 #16
EPOCHS = 0 #1600
# width, height = int(1536.959604286892), int(82.0964550700742)
# width, height = 2700,90
# width, height = 512,512
width, height = 1024,1024
exp_name = f"exp_{corpus}_{model}_size{width}"
learning_rate = 0.001 # 0.0005
momentum = 0
num_input_channels=3
k_steps=1
opts=None
work_dir = f"work_{corpus}_{model}"

device = torch.device("cuda:{}".format(gpu - 1) if gpu else "cpu")

logger_csv = CSVLogger(work_dir, name=exp_name)
path_save = os.path.join(work_dir, "checkpoints")

imgDataset = ImageDataset(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=corpus)


net = Net(  num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                 learning_rate=learning_rate, n_classes=imgDataset.n_classes,momentum=momentum, milestones=steps, model=model
           )

net = net.load_from_checkpoint(checkpoint_load, num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                learning_rate=learning_rate, n_classes=imgDataset.n_classes, model=model)
net.to(device)

trainer = pl.Trainer(min_epochs=EPOCHS, max_epochs=EPOCHS, logger=[logger_csv], #wandb_logger
                default_root_dir=path_save,
                gpus=gpu,
                log_every_n_steps=k_steps
            )
if do_train:
    trainer.fit(net, imgDataset)

test_loader = imgDataset.test_dataloader()
batch = next(iter(test_loader))
images = batch['pixel_values']

background = images.to(device)
net = net.to(device)
print("Deep Explainer")
e = shap.DeepExplainer(net, background)

# batch = next(iter(test_loader))
# images = batch['pixel_values']
print("Getting shap values")
shap_values = e.shap_values(images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(images.cpu().numpy(), 1, -1), 1, 2)

import matplotlib.pyplot as plt
# f = plt.gcf()
shap.image_plot(shap_numpy, -test_numpy, show=False)

plt.savefig('scratch.png')