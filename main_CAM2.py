
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
import cv2
from torch.nn import functional as F

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
batch_size = 1 #16
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

net.eval()
# Preprocess your data and feed it to the model
image = images[0]
# out = net(image)

finalconv_name = 'ResNetStage'


# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
# print(net._modules.get('model'))
print(net._modules.get('model').encoder.stages[-1].layers[-1].layer[-1].convolution)
net._modules.get('model').encoder.stages[-1].layers[-1].layer[-1].convolution.register_forward_hook(hook_feature)



# get the softmax weight
params = list(net.parameters())
# print("----> ", params)
weight_softmax = np.squeeze(params[-1].data.numpy())

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (1024,1024)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        print(weight_softmax[idx].shape)
        print(feature_conv.shape)
        print(feature_conv.reshape((nc, h*w)).shape)
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

logit = net(images)[0]
print(logit)

classes = imgDataset.tag2label

h_x = torch.exp(logit).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

print(probs, idx)
# output the prediction
print(probs[0])
print(classes)
for i in range(0, 3):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread('test.jpg')
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)