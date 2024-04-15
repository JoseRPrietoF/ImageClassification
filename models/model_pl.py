from logging.config import valid_ident
from sklearn.model_selection import validation_curve
import torch.optim as optim
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .operations import ConvBlock
import torchmetrics
from transformers import AdamW, ResNetModel, SwinForImageClassification, ConvNextForImageClassification
import torchvision.models as models
from torch import Tensor
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomBrightness, RandomThinPlateSpline, RandomContrast, RandomSaturation, RandomAffine

class debug(nn.Module):

    def forward(self, x):
        print(x.shape)
        return x
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            RandomChannelShuffle(p=0.75),
            RandomThinPlateSpline(p=0.75),
            RandomBrightness(p=0.75),
            RandomContrast(p=0.75),
            RandomSaturation(p=0.75),
            RandomAffine(p=0.75, degrees=(10,10), translate=(0.10,0.10))
        )

        self.jitter = ColorJitter(0.7, 0.5, 0.5, 0.5)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class Net(pl.LightningModule):
    def __init__(self, 
                num_input_channels,
                 opts,
                 width, height,
                 learning_rate=0.001,
                 n_classes=3,
                 momentum=0,
                 milestones=[5000,10000,15000,16000,18000],
                 model="resnet50",
                torchvision = False,
                layers=[128,128],
                len_feats=1024,
                DO_text=0.3):
        super(Net, self).__init__()


        # dataset = load_dataset("huggingface/cats-image")
        # image = dataset["test"]["image"][0]
        w, h = width, height
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.milestones = milestones
        self.model_selected = model
        self.torchvision = torchvision
        self.transform = DataAugmentation()  # per batch augmentation_kornia
        
        if "fusion" in model:
            model_text = [nn.Linear((len_feats), layers[0]),
                    nn.BatchNorm1d(layers[0]),
                    nn.ReLU(True),
                    nn.Dropout(DO_text)
                    ]
            for i in range(1, len(layers)):
                model_text = model_text + [nn.Linear(layers[i-1], layers[i]),
                    nn.BatchNorm1d(layers[i]),
                    nn.ReLU(True),
                    nn.Dropout(DO_text)
                    ]
            self.model_text = nn.Sequential(*model_text)
            print(self.model_text)

        if "resnet" in model:
            if "resnet50" in model:
                if torchvision:
                    self.model = models.resnet50(pretrained=True)
                else:
                    self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
            elif "resnet18" in model:
                # self.model = models.resnet18(pretrained=True)
                self.model = ResNetModel.from_pretrained("microsoft/resnet-18")
            elif "resnet101" in model:
                # self.model = models.resnet101(pretrained=True)
                self.model = ResNetModel.from_pretrained("microsoft/resnet-101")
            # print(self.model.layer4[2].conv3)
            if torchvision:
                dim_out_last_channel = self.model.layer4[2].conv3.out_channels
            else:
                dim_out_last_channel = self.model.encoder.stages[-1].layers[-1].layer[-1].convolution.out_channels
            # conv = nn.Conv2d(dim_out_last_channel, dim_out_last_channel//4, kernel_size=4, stride=2, padding=1)
            conv = ConvBlock(dim_out_last_channel, dim_out_last_channel//4, kernel_size=3, stride=1, poolsize=(2,2), batchnorm=True, activation=nn.ReLU)
            dim_out_last_channel = dim_out_last_channel//4
            # conv2 = nn.Conv2d(dim_out_last_channel, dim_out_last_channel//4, kernel_size=4, stride=2, padding=1)
            conv2 = ConvBlock(dim_out_last_channel, dim_out_last_channel//4, kernel_size=3, stride=1, poolsize=(2,2), batchnorm=True, activation=nn.ReLU)
            dim_out_last_channel = dim_out_last_channel//4
            
            steps = 5 + 2
            new_w, new_h = (w // (2**steps)), (h // (2**steps))
            new_shape = new_h * new_w * dim_out_last_channel
            if torchvision:
                linear = nn.Linear(new_shape, n_classes)
                print(w,h, new_w, new_h,dim_out_last_channel, new_shape )
                # self.model = nn.Sequential(*[model, self.linear])
                # self.model_clasif = nn.Sequential(*[  conv, conv2, nn.Flatten(), linear])
                # self.model.fc = Identity()
                self.model.avgpool = nn.Sequential(*[  conv, conv2])
                print(self.model)
                # print(self.model_clasif)
                self.model.fc = linear
            else:
                if "fusion" in model:
                    # linear = nn.Linear(new_shape, layers[-1])
                    linear_text = nn.Sequential(*[nn.Linear(new_shape, layers[-1]),
                        nn.BatchNorm1d(layers[-1]),
                        nn.ReLU(True),
                        nn.Dropout(DO_text)
                    ])
                    
                    linear_fusion2_text = [nn.Linear( layers[-1], layers[-1]),
                        nn.BatchNorm1d(layers[-1]),
                        nn.ReLU(True),
                        nn.Dropout(DO_text),
                        nn.Linear(layers[-1], n_classes)
                    ]
                    # linear_fusion2 = linear_fusion2 + [nn.Linear(layers[-1], n_classes)]
                    self.linear_fusion = nn.Sequential(*linear_fusion2_text)
                    self.model_clasif = nn.Sequential(*[  conv, conv2, nn.Flatten(), linear_text])
                else:
                    linear = nn.Linear(new_shape, n_classes)
                    self.model_clasif = nn.Sequential(*[  conv, conv2, nn.Flatten(), linear])
                # self.model_clasif = nn.Sequential(*[ debug(), conv, debug(), conv2, debug(), nn.Flatten(), debug(), linear])
                    print(self.model_clasif)
        elif "swintiny" in model:
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
            self.model.classifier = nn.Linear(768, n_classes)
            print(self.model)
        elif "swinbase" in model:
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224")
            self.model.classifier = nn.Linear(1024, n_classes)
            print(self.model)
        elif "swinv2_base" in model:
            # self.model = SwinForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window16-256")
            self.model = SwinForImageClassification.from_pretrained("microsoft/swinv2-base-patch4-window12to24-192to384-22kto1k-ft")
            self.model.classifier = nn.Linear(1024, n_classes)
            print(self.model)
        elif "convnext" in model:
            if "large" in model:
                self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-large-224")
                self.model.classifier = nn.Linear(768, n_classes)
            elif "tiny" in model:
                self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224")
                self.model.classifier = nn.Linear(768, n_classes)
            elif "base" in model:
                self.model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-384")
                self.model.classifier = nn.Linear(1024, n_classes)
            
            print(self.model)

        
        # inputs = feature_extractor(image, return_tensors="pt")

        # with torch.no_grad():
        #     logits = model(**inputs).logits

        # # model predicts one of the 1000 ImageNet classes
        # predicted_label = logits.argmax(-1).item()
        # print(model.config.id2label[predicted_label])


        # self.criterion = torch.nn.CTCLoss()
        self.criterion = torch.nn.NLLLoss()
        self.m = nn.LogSoftmax(dim=-1)
     
        # print(self.net)
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.test_acc = torchmetrics.Accuracy()
        # self.automatic_optimization = False
        # print("self.automatic_optimization", self.automatic_optimization)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training:
            batch["pixel_values"] = self.transform(batch["pixel_values"])  # => we perform GPU/Batched data augmentation
        return batch

    def forward(self, batch):
        if "fusion" in self.model_selected:
            x_text = batch['text']
            # x_text = torch.zeros_like(x_text)
            x_text = self.model_text(x_text)
        try:
            x = batch['pixel_values']
        except:
            x = batch
        x = self.model(x)
        if "resnet" in self.model_selected and not self.torchvision:
            x = self.model_clasif(x.last_hidden_state)
            # x = self.model_clasif(x)
        else:
            x = x.logits

        if "fusion" in self.model_selected:
            x = self.linear_fusion(x + x_text)

        x = F.log_softmax(x, dim=-1)
        # exit()
        return x
    
    def configure_optimizers(self):
        def is_text(n): return 'text' in n
        params = list(self.named_parameters())
        grouped_parameters = [
            {"params": [p for n, p in params if is_text(n)], 'lr': self.learning_rate},
            {"params": [p for n, p in params if not is_text(n)], 'lr': self.learning_rate},
        ]
        optimizer=AdamW(
            grouped_parameters, lr=self.learning_rate
            )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.5)
        return [optimizer], [scheduler]
    
  
    def training_step(self, train_batch, batch_idx):
        # get the inputs
        
        hyp = self(train_batch)
        hyp = self.m(hyp)
        loss = self.criterion(hyp, train_batch['labels'])
     
        self.log('train_loss', loss)

        self.train_acc(torch.exp(hyp), train_batch['labels'])
        self.log('train_acc_step', self.train_acc)
      
        return {'loss': loss, 'outputs': hyp, 'y_gt':train_batch['labels']}

    def training_epoch_end(self, outs):
        # log epoch metric
        print("\n   TRAIN\n")
        loss = np.mean([o['loss'].item() for o in outs])
        self.log('train_loss_epoch', loss)
        print(f'Train loss {loss}')

        self.log('train_acc_epoch', self.train_acc)
        outputs = []
        gts = []
        for x in outs:
            o = x['outputs']
            # print(o.shape)
            outputs.extend(torch.argmax(torch.exp(o), dim=-1))
            gts.extend(x['y_gt']) 
        
        outputs = torch.Tensor(outputs)   
        gts = torch.Tensor(gts)   
        acc = (outputs == gts).sum() / gts.size(0)
        print(f'Accuracy train: {acc}')

    
    def validation_step(self, val_batch, batch_idx):

        hyp = self(val_batch)
        hyp = self.m(hyp)
        loss = self.criterion(hyp,val_batch['labels'])
        self.log('val_loss', loss)

        self.val_acc(torch.exp(hyp), val_batch['labels'])
        self.log('val_acc_step', self.val_acc)
      
        return {'loss': loss, 'outputs': hyp, 'y_gt':val_batch['labels']}
    
    def validation_epoch_end(self, outs):
        # log epoch metric
        print("\n   VAL\n")

        loss = np.mean([o['loss'].item() for o in outs])
 
        self.log('val_loss_epoch', loss)
        print(f'Val loss {loss}')

        self.log('val_acc_epoch', self.val_acc)
        print("self.val_acc ", self.val_acc)
        outputs = []
        gts = []
        for x in outs:
            o = x['outputs']
            # print(o.shape)
            outputs.extend(torch.argmax(torch.exp(o), dim=-1))
            gts.extend(x['y_gt']) 
        
        outputs = torch.Tensor(outputs)   
        gts = torch.Tensor(gts)   
        acc = (outputs == gts).sum() / gts.size(0)
        print(f'Accuracy val: {acc}')
    
    def test_step(self, test_batch, batch_idx):
        # get the inputs
        hyp = self(test_batch)
        hyp = self.m(hyp)
        
        loss = self.criterion(hyp,test_batch['labels'])
        self.log('test_loss', loss)

        self.test_acc(hyp, test_batch['labels'])
        self.log('test_acc_step', self.test_acc)

        return {'loss': loss, 'outputs': hyp, 'y_gt':test_batch['labels']}

    
    def test_epoch_end(self, outs):
        # log epoch metric
        print("\n  TEST ")
        loss = np.mean([o['loss'].item() for o in outs])
        print(f'Test loss {loss}')

        outputs = []
        gts = []
        for x in outs:
            o = x['outputs']
            # print(o.shape)
            outputs.extend(torch.argmax(torch.exp(o), dim=-1))
            gts.extend(x['y_gt']) 
        
        outputs = torch.Tensor(outputs)   
        gts = torch.Tensor(gts)   
        acc = (outputs == gts).sum() / gts.size(0)
        print(f'Accuracy test: {acc}')

        return outs

    
    def predict_step(self, train_batch, batch_idx):
        # get the inputs
        # filename = train_batch['filenames']
        filename=""
        hyp = self(train_batch)
        hyp = self.m(hyp)
        outputs = torch.exp(hyp).cpu().detach().numpy()
        # print("Predicting image ", filename, outputs.shape)
        return {'outputs':outputs, "filenames":filename}
    
    def check(self, x, inp="x"):
        if x is not None:
            if torch.sum(torch.isnan(x)).item() > 0:
                raise ValueError("NaN values in {}".format(inp))
            if torch.sum(torch.isinf(x)).item() > 0:
                raise ValueError("+/-Inf in {}".format(inp))
    
    @staticmethod
    def get_conv_output_size(
            size,  # type: Tuple[int, int]
            cnn_kernel_size,  # type: Sequence[Union[int, Tuple[int, int]]]
            cnn_stride,  # type: Sequence[Union[int, Tuple[int, int]]]
            cnn_dilation,  # type: Sequence[Union[int, Tuple[int, int]]]
            cnn_poolsize,  # type: Sequence[Union[int, Tuple[int, int]]]
    ):
        size_h, size_w = size
        for ks, st, di, ps in zip(
                cnn_kernel_size, cnn_stride, cnn_dilation, cnn_poolsize
        ):
            if type(ks) == int:
                ks = (ks, ks)
            if type(st) == int:
                st = (st, st)
            if type(di) == int:
                di = (di, di)
            size_h = ConvBlock.get_output_size(
                size_h, kernel_size=ks[0], dilation=di[0], stride=st[0], poolsize=ps[0]
            )
            size_w = ConvBlock.get_output_size(
                size_w, kernel_size=ks[1], dilation=di[1], stride=st[1], poolsize=ps[1]
            )
        return size_h, size_w

