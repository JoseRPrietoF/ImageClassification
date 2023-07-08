
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
import sys
import argparse
from main import parse_args

def main(args):
    corpus = args.corpus
    img_dirs = args.img_dirs
    prod = False
    output_name = args.output_name

    steps=[10]
    model = args.model
    # model = "convnext_base"
    # if "JMBD" in corpus:
    #     checkpoint_load = f"works/1folder/work_{corpus}_LOO_{model}_size1024/checkpoints/*/*/*/*ckpt"
    # else:
    #     checkpoint_load = f"works/2folder/work_JMBD_tr_{corpus}_{model}_size1024/checkpoints/*/*/*/*ckpt"
    checkpoint_load = args.checkpoint_load
    checkpoint_load = glob.glob(checkpoint_load)[0]
    txt_path_tr=args.txt_path_tr
    txt_path_te=args.txt_path_te
    feats=args.feats
    layers=args.layers
    # folder = corpus
    # img_dirs = "/home/jose/projects/image_classif/data/{}/prod_{}".format("JMBD4949_4950", folder) # _4946_4952
    # corpus = f"JMBD_{corpus}_prod"

    gpu = args.gpu
    batch_size = args.batch_size #16
    EPOCHS = args.epochs #15 for resnet
    width, height = args.width, args.width
    exp_name = args.exp_name
    learning_rate = args.learning_rate # resnet
    # learning_rate = 0.01 # 0.0005
    momentum = 0
    num_input_channels = args.num_input_channels
    k_steps= args.log_every_n_steps
    opts=None
    str_layers = "_".join([str(x) for x in layers])
    # work_dir = f"work_{corpus}_{model}_size{width}_{str_layers}_v3"
    # if "JMBD" in corpus:
    #     work_dir = f"works/1folder/work_{corpus}_LOO_{model}_size{width}"
    # else:
    #     work_dir = f"works/2folder/work_JMBD_tr_{corpus}_{model}_size{width}"
    # print(work_dir)
    # exit()
    work_dir = args.work_dir

    device = torch.device("cuda:{}".format(gpu - 1) if gpu else "cpu")

    logger_csv = CSVLogger(work_dir, name=exp_name)
    path_save = os.path.join(work_dir, "checkpoints")

    # imgDataset_trained = ImageDataset(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=corpus, nfeats=feats, txt_path_tr=txt_path_tr, txt_path_te=txt_path_te)

    imgDataset = ImageDatasetProd(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=output_name, nfeats=feats, txt_path_tr=txt_path_tr, txt_path_te=txt_path_te, n_classes=args.n_classes)

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
    fname_file = os.path.join(work_dir, f"results_{output_name}")
    fname_file_errors = os.path.join(work_dir, f"results_errors_{output_name}")
    save_to_file(results, imgDataset.tag2label, fname_file, fname_file_errors, prod=prod)
    # for fname, output in results:
    #     print(fname, output)

if __name__ == "__main__":
    args = parse_args()
    main(args)