from logging.config import valid_ident
import torch.optim as optim
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .operations import ConvBlock
import torchmetrics
from transformers import AdamW, ResNetModel

class debug(nn.Module):

    def forward(self, x):
        print(x.shape)
        return x

class Net(pl.LightningModule):
    def __init__(self, 
                num_input_channels,
                 opts,
                 width, height,
                 learning_rate=0.001,
                 n_classes=3,
                 momentum=0,
                 milestones=[5000,10000,15000,16000,18000]):
        super(Net, self).__init__()


        # dataset = load_dataset("huggingface/cats-image")
        # image = dataset["test"]["image"][0]
        w, h = width, height
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.milestones = milestones
        
        self.model = ResNetModel.from_pretrained("microsoft/resnet-50")
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
        linear = nn.Linear(new_shape, n_classes)
        print(w,h, new_w, new_h,dim_out_last_channel, new_shape )
        # self.model = nn.Sequential(*[model, self.linear])
        self.model_clasif = nn.Sequential(*[  conv, conv2, nn.Flatten(), linear])
        print(self.model_clasif)
        # print(self.model)

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



    def forward(self, x):
        x = self.model(x).last_hidden_state
        x = self.model_clasif(x)
        x = F.log_softmax(x, dim=-1)
        # exit()
        return x
    
    def configure_optimizers(self):
        optimizer=AdamW(
            self.parameters(), lr=self.learning_rate
            )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=0.5)
        return [optimizer], [scheduler]
    
  
    def training_step(self, train_batch, batch_idx):
        # get the inputs
        hyp = self(train_batch['pixel_values'])
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
        print(batch_idx)
        hyp = self(val_batch['pixel_values'])
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
        hyp = self(test_batch['pixel_values'])
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
        hyp = self(train_batch['pixel_values'])
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
