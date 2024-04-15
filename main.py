import argparse
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
# from models.operations import save_file
# from models import model
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import sys
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

def main(args):

    steps=args.milestones
    seed = 1
    # work_JMBD4949_4950_256feats_1example_rnnunits_64_rnnlayers3_cnnFalse_RGB/checkpoints/exp1_exp1/0_1oguiho2/checkpoints/epoch=18512-step=18512.ckpt
    # checkpoint_load = "work_JMBD4949_4950_1024feats_rnnunits_256_rnnlayers3_cnnFalse_RGB_relative_split_notForced_noBI/checkpoints/exp1_exp1/0_edjf8iut/checkpoints/epoch=499-step=63999.ckpt"
    checkpoint_load = args.checkpoint_load
    do_train = args.do_train
    model = args.model
    # model = "resnet50"
    # model = "swinbase"
    txt_path_tr=args.txt_path_tr
    txt_path_te=args.txt_path_te
    feats=args.feats
    layers=args.layers
    # txt_path_tr="/data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/tfidf_tr49.txt" 
    # txt_path_te="/data2/jose/projects/docClasifIbPRIA22/data/JMBD4949_4950/IG_TFIDF/tr49/tfidf_te50.txt"
    # model = f"resnet50fusion{feats}feats"

    img_dirs = args.img_dirs
    # corpus = f"JMBD4949_4950_prod"
    corpus = args.corpus


    gpu = args.gpu
    batch_size = args.batch_size #16
    EPOCHS = args.epochs #15 for resnet

    # width, height = args.width, args.height
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
    # if "SiSe" in corpus:
    #     work_dir = f"works/SiSe/work_{model}_size{width}"
    # elif "JMBD" in corpus:
    #     work_dir = f"works/1folder/work_{corpus}_{model}_size{width}"
    # else:
    #     work_dir = f"works/2folder/work_{corpus}_{model}_size{width}"
    work_dir = args.work_dir
    # print(work_dir)
    # exit()

    device = torch.device("cuda:{}".format(gpu - 1) if gpu else "cpu")

    logger_csv = CSVLogger(work_dir, name=exp_name)
    # wandb_logger = WandbLogger(project=exp_name)
    path_save = os.path.join(work_dir, "checkpoints")
    print(f"img_dirs {img_dirs}")
    early_stop_callback = EarlyStopping(monitor="val_acc_epoch", min_delta=0.00, patience=args.patience, verbose=False, mode="max")
    imgDataset = ImageDataset(batch_size=batch_size, width=width, height=height, nchannels=num_input_channels, work_dir=work_dir, img_dirs=img_dirs, corpus=corpus, nfeats=feats, txt_path_tr=txt_path_tr, txt_path_te=txt_path_te, n_classes=args.n_classes)


    net = Net(  num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                    learning_rate=learning_rate, n_classes=imgDataset.n_classes,momentum=momentum, milestones=steps, model=model, layers=layers, len_feats=feats
            )
    if checkpoint_load:
        net = net.load_from_checkpoint(checkpoint_load, num_input_channels=num_input_channels,opts=opts,width=width, height=height,
                    learning_rate=learning_rate, n_classes=imgDataset.n_classes, model=model, strict=False)
    net.to(device)
    # wandb_logger.watch(net)
    trainer = pl.Trainer(min_epochs=EPOCHS, max_epochs=EPOCHS, logger=[logger_csv], #wandb_logger
                    default_root_dir=path_save,
                    gpus=gpu,
                    log_every_n_steps=k_steps,
                    auto_lr_find=args.auto_lr_find,
                    callbacks=[early_stop_callback]
                )
    if do_train:
        trainer.fit(net, imgDataset)


    ##TEST
    print("TEST")
    results_test = trainer.test(net, imgDataset)
    # print("results_test   ", results_test)
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
    save_to_file(results, imgDataset.tag2label, fname_file, fname_file_errors)
    # for fname, output in results:
    #     print(fname, output)

def parse_args():
    parser = argparse.ArgumentParser(description='Get the sequence')
    parser.add_argument('--checkpoint_load', type=str, help='model', default="")
    parser.add_argument('--do_train', type=str, help='folder', default="true")
    parser.add_argument('--model', type=str, help='folder', default="")
    parser.add_argument('--txt_path_tr', type=str, help='algorithm', default=None)
    parser.add_argument('--txt_path_te', type=str, help='algorithm', default=None)
    parser.add_argument('--width', type=int, help='algorithm', default=1024)
    parser.add_argument('--height', type=int, help='algorithm', default=1024)
    parser.add_argument('--num_input_channels', type=int, help='algorithm', default=3)
    parser.add_argument('--learning_rate', type=float, help='algorithm', default=0.001)
    parser.add_argument('--feats', type=int, help='algorithm', default=1024)
    parser.add_argument('--batch_size', type=int, help='algorithm', default=4)
    parser.add_argument('--epochs', type=int, help='algorithm', default=50)
    parser.add_argument('--gpu', type=int, help='algorithm', default=1)
    parser.add_argument('--layers', type=list, help='algorithm', default=[64,64])
    parser.add_argument('--milestones', type=list, help='algorithm', default=[10])
    parser.add_argument('--corpus', type=str, help='algorithm')
    parser.add_argument('--log_every_n_steps', type=int, help='algorithm', default=1)
    parser.add_argument('--exp_name', type=str, help='algorithm', default="work")
    parser.add_argument('--work_dir', type=str, help='algorithm', default="work")
    parser.add_argument('--img_dirs', type=str, help='algorithm', default="/home/jose/projects/image_classif/data/")
    parser.add_argument('--n_classes', type=int, help='algorithm', default=3)
    parser.add_argument('--output_name', type=str, help='model', default="")
    parser.add_argument('--auto_lr_find', type=str, help='model', default="no")
    parser.add_argument('--patience', type=int, help='model', default=10)
    parser.add_argument('--da', type=str, help='data augmentation', default="true")
    args = parser.parse_args()
    args.do_train = args.do_train.lower() in ["si", "true", "yes"]
    args.auto_lr_find = args.auto_lr_find.lower() in ["si", "true", "yes"]
    args.da = args.da.lower() in ["si", "true", "yes"]
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)