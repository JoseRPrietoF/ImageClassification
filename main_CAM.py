
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
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image

steps=[10]
seed = 1
# work_JMBD4949_4950_256feats_1example_rnnunits_64_rnnlayers3_cnnFalse_RGB/checkpoints/exp1_exp1/0_1oguiho2/checkpoints/epoch=18512-step=18512.ckpt
# checkpoint_load = "work_JMBD4949_4950_1024feats_rnnunits_256_rnnlayers3_cnnFalse_RGB_relative_split_notForced_noBI/checkpoints/exp1_exp1/0_edjf8iut/checkpoints/epoch=499-step=63999.ckpt"
# checkpoint_load = "/data2/jose/projects/image_classif/work_JMBD4949_4950_tr49_resnet50/checkpoints/exp_JMBD4949_4950_tr49_resnet50_size1024_exp_JMBD4949_4950_tr49_resnet50_size1024/3_2qdjex9p/checkpoints/epoch=11-step=3299.ckpt"
# checkpoint_load = "/data2/jose/projects/image_classif/work_hisclima_resnet50_size2562/checkpoints/exp_hisclima_resnet50_exp_hisclima_resnet50/0_ayr064k3/checkpoints/epoch=14-step=584.ckpt"
checkpoint_load = "/home/jose/projects/image_classif/works/AHPC_encabezados/work_resnet50_size512/checkpoints/work/0/checkpoints/epoch=29-step=2639.ckpt"
do_train = True
model = "resnet50"

# corpus = f"hisclima"
# img_dirs = f"/home/jose/projects/image_classif/data/Hisclima"

tr_="tr49"
# corpus = f"JMBD4949_4950_{tr_}"
corpus = f"hisclima"
# img_dirs = f"/home/jose/projects/image_classif/data/JMBD4949_4950/{tr_}"
# img_dirs = f"/home/jose/projects/image_classif/data/Hisclima"
img_dirs = f"/home/jose/projects/image_classif/data/AHPC_cabeceras"

# corpus = f"JMBD4949"
# img_dirs = "/home/jose/projects/image_classif/data/{}".format(corpus)

gpu = 1
batch_size = 1 #16
EPOCHS = 0 #1600
# width, height = int(1536.959604286892), int(82.0964550700742)
# width, height = 2700,90
# width, height = 512,512
width, height = 512,512
exp_name = f"exp_{corpus}_{model}_size{width}"
learning_rate = 0.001 # 0.0005
momentum = 0
num_input_channels=3
n_classes = 12
k_steps=1
opts=None
# work_dir = f"work_{corpus}_{model}"
# work_dir = "work_JMBD4949_4950_tr49_resnet50"
work_dir = f"/home/jose/projects/image_classif/works/AHPC_encabezados/work_resnet50_size512/"

device = torch.device("cuda:{}".format(gpu - 1) if gpu else "cpu")

logger_csv = CSVLogger(work_dir, name=exp_name)
path_save = os.path.join(work_dir, "checkpoints")

imgDataset = ImageDataset(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=corpus, n_classes=n_classes)

feats=1024
layers=[64,64]
# net = Net(  num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                #  learning_rate=learning_rate, n_classes=imgDataset.n_classes,momentum=momentum, milestones=steps, model=model, torchvision=False
        #    )
net = Net(  num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                    learning_rate=learning_rate, n_classes=imgDataset.n_classes,momentum=momentum, milestones=steps, model=model, layers=layers, len_feats=feats
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
# batch = next(iter(test_loader))
# images = batch['pixel_values']
fnames = [i['image'].filename for i in imgDataset.dataset['test']]
# print(net)
net.eval()
named_layers = dict(net.named_modules())

layer = 'model.encoder.stages.3.layers.2.layer.2.convolution'
#  layer = "model_clasif.1.activation"
cam_extractor = SmoothGradCAMpp(net, layer) #'layer4'
# Preprocess your data and feed it to the model

for l in named_layers:
    print(l)

path_save_imgs = os.path.join(work_dir, "explainable")
if not os.path.exists(path_save_imgs):
    os.mkdir(path_save_imgs)

for i, batch in enumerate(test_loader):
    images = batch['pixel_values']

    fname = fnames[i]
    print(fname)
    fname = fname.split("/")[-1].split(".")[0]



    image = images[0]
    out = net(images)
    out = torch.exp(out)
    print(out)
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


    # Resize the CAM and overlay it
    result = overlay_mask(to_pil_image(image), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.9)
    # Display it
    plt.imshow(result)
    plt.axis('off')
    plt.tight_layout()
    # plt.show()
    p = os.path.join(path_save_imgs, f"{fname}.jpg")
    plt.savefig(p)