
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
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from data import ImageDataset, ImageDatasetProd
from models import Net
from torch import nn, save
from utils.functions import save_to_file
import glob
# from models.operations import save_file
# from models import model
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

steps=[10]
seed = 1
model = "convnext_base"
checkpoint_load = f"work_JMBD4949_4950_prod_{model}_size1024/checkpoints/exp_JMBD4949_4950_prod_{model}_exp_JMBD4949_4950_prod_{model}/*/*/*ckpt"
checkpoint_load = glob.glob(checkpoint_load)[0]
txt_path_tr=None
txt_path_te=None
feats=1024
layers=[64,64]
folder = "4952"
img_dirs = "/home/jose/projects/image_classif/data/{}/prod_{}".format("JMBD4949_4950", folder) # _4946_4952
corpus = f"JMBD_{folder}_prod"

gpu = 1
batch_size = 5 #16
EPOCHS = 15 #1600
width, height = 1024,1024
exp_name = f"exp_{corpus}_{model}"
learning_rate = 0.001 # 0.0005
momentum = 0
num_input_channels=3
k_steps=1
opts=None
str_layers = "_".join([str(x) for x in layers])
# work_dir = f"work_{corpus}_{model}_size{width}_{str_layers}_v3"
work_dir = f"work_{corpus}_{model}_size{width}"
# print(work_dir)
# exit()

device = torch.device("cuda:{}".format(gpu - 1) if gpu else "cpu")

logger_csv = CSVLogger(work_dir, name=exp_name)
path_save = os.path.join(work_dir, "checkpoints")

# imgDataset_trained = ImageDataset(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=corpus, nfeats=feats, txt_path_tr=txt_path_tr, txt_path_te=txt_path_te)

imgDataset = ImageDatasetProd(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=corpus, nfeats=feats, txt_path_tr=txt_path_tr, txt_path_te=txt_path_te)


net = Net(  num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                 learning_rate=learning_rate, n_classes=imgDataset.n_classes,momentum=momentum, milestones=steps, model=model, layers=layers, len_feats=feats
           )

if checkpoint_load:
    net = net.load_from_checkpoint(checkpoint_load, num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                 learning_rate=learning_rate, n_classes=imgDataset.n_classes, model=model)
net.to(device)
trainer = pl.Trainer(min_epochs=EPOCHS, max_epochs=EPOCHS, logger=[logger_csv], #wandb_logger
                default_root_dir=path_save,
                gpus=gpu,
                log_every_n_steps=k_steps
            )

##TEST
print("PREDICT")
outputs = trainer.predict(net, imgDataset) # , ckpt_path='best'
outputs_ = [x['outputs'] for x in outputs]
outputs = []
for o in outputs_:
    outputs.extend(o)
fnames = [i['image'].filename for i in imgDataset.dataset['test']]
print(len(fnames))
print(len(outputs))
results = zip(fnames, outputs)
fname_file = os.path.join(work_dir, "results")
fname_file_errors = os.path.join(work_dir, "results_errors")
save_to_file(results, imgDataset.tag2label, fname_file, fname_file_errors, prod=True)
# for fname, output in results:
#     print(fname, output)
