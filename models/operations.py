from numpy import dtype
import torch, re
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Optional, Tuple, Union
import math
import pickle

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,  # type: int
        out_channels,  # type: int
        kernel_size=3,  # type: Union[int, Tuple[int, int]]
        stride=1,  # type: Union[int, Tuple[int, int]]
        dilation=1,  # type: Union[int, Tuple[int, int]]
        activation=nn.LeakyReLU,  # type: Optional[nn.Module]
        poolsize=None,  # type: Optional[Union[int, Tuple[int, int]]]
        dropout=None,  # type: Optional[float]
        batchnorm=False,  # type: bool
    ):
        # type: (...) -> None
        super(ConvBlock, self).__init__()

        self.dropout = dropout
        self.in_channels = in_channels
        self.poolsize = poolsize

        if not isinstance(kernel_size, (list, tuple)):
            kernel_size = (kernel_size, kernel_size)
        if not isinstance(dilation, (list, tuple)):
            dilation = (dilation,) * 2
        # Add Conv2d layer (compute padding to perform a full convolution).
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=tuple(
                (kernel_size[dim] - 1) // 2 * dilation[dim] for dim in (0, 1)
            ),
            dilation=dilation,
            # Note: If batchnorm is used, the bias does not affect the output
            # of the unit.
            bias=not batchnorm,
        )

        # Add Batch normalization
        self.batchnorm = nn.BatchNorm2d(out_channels) if batchnorm else None

        # Activation function must support inplace operations.
        self.activation = activation(inplace=False) if activation else None

        # Add maxpool layer
        self.pool = nn.MaxPool2d(poolsize) if self.poolsize and self.poolsize[0]>0 and \
        self.poolsize[1] > 0 else None
    
    @staticmethod
    def get_output_size(
        size: Union[torch.Tensor, int],
        kernel_size: int,
        dilation: int,
        stride: int,
        poolsize: int,
        padding: Optional[int] = None,
    ) -> Union[torch.LongTensor, int]:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        size = size.float() if isinstance(size, torch.Tensor) else float(size)
        size = (size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
        size = size.floor() if isinstance(size, torch.Tensor) else math.floor(size)
        if poolsize:
            size /= poolsize
        return (
            size.floor().long()
            if isinstance(size, torch.Tensor)
            else int(math.floor(size))
        )

    def forward(self, x):
        # type: (Union[Tensor, PaddedTensor]) -> Union[Tensor, PaddedTensor]
        # print(x)
        # print(PaddedTensor)
        # x, xs = (x.data, x.sizes) if isinstance(x, PaddedTensor) else (x, None)
        # print(x)
        # x, xs = (x.data, x.sizes)
        xs = None
        assert x.size(1) == self.in_channels, (
            "Input image depth ({}) does not match the "
            "expected ({})".format(x.size(1), self.in_channels)
        )

        if self.dropout and 0.0 < self.dropout < 1.0:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv(x)
        # if self.use_masks:
        #     x = mask_image_from_size(x, batch_sizes=xs, mask_value=0)

        if self.batchnorm:
            x = self.batchnorm(x)

        if self.activation:
            x = self.activation(x)

        if self.pool:
            x = self.pool(x)
        return x
        # return (
        #     x if xs is None else PaddedTensor.build(x, self.get_batch_output_size(xs))
        # )


class ImagePoolingSequencer(torch.nn.Module):
    def __init__(self, sequencer=None, columnwise=True, _fix_size=5):
        super(ImagePoolingSequencer, self).__init__()

        self._fix_size = _fix_size
        self._columnwise = columnwise
        #self._fix_size = int(m.group(2))

        # Assume that the images have a fixed height
        # (or width if columnwise=False)
        self.sequencer = None

    @property
    def columnwise(self):
        return self._columnwise

    @property
    def fix_size(self):
        return self._fix_size

    def forward(self, x):
        if self._fix_size is not None:
            if self._columnwise and x.size(-2) != self._fix_size:
                raise ValueError(
                    "Input images must have a fixed height of {} pixels, "
                    "size is {} - height is {}".format(self._fix_size, str(x.size()), str(x.size(-2)))
                )
            elif not self._columnwise and x.size(-1) != self._fix_size:
                raise ValueError(
                    "Input images must have a fixed width of {} pixels, "
                    "size is {}".format(self._fix_size, str(x.size()))
                )

        x = image_to_sequence(x, columnwise=self._columnwise, return_packed=True)
        return x

def image_to_sequence(x, columnwise=True, return_packed=False):
    x, xs = (x, None)

    if x.dim() == 2:
        x = x.view(1, 1, x.size(0), x.size(1))
    elif x.dim() == 3:
        x = x.view(1, x.size(0), x.size(1), x.size(2))
    assert x.dim() == 4

    n, c, h, w = x.size()
    if columnwise: # por defecto. HTR.
        x = x.permute(3, 0, 1, 2).contiguous().view(w, n, h * c)
    else:
        x = x.permute(2, 0, 1, 3).contiguous().view(h, n, w * c)

    return x

def calculate_input_lengths(input_lengths):
    n = []
    # for i in input_lengths:
        # n.append(i.shape[0])
    # for i in range(input_lengths.shape[1]):
    #     n.append(input_lengths.shape[0])
    input_lengths = torch.full(size=(input_lengths.shape[1],), fill_value=input_lengths.shape[0], dtype=torch.long)
    return input_lengths

class Collator(object):
    
    def __call__(self, batch):
        # item = {'img': batch['img'], 'idx':indexes}
        # if 'label' in batch[0].keys():
        #     labels = [item['label'] for item in batch]
        #     item['label'] = labels
        imgs = []
        ids = []
        txts = []
        target_lengths = []
        for b in batch:
            imgs.append(b['img'])
            ids.append(b['id'])
            txts.extend(b['txt'])
            target_lengths.append(len(b['txt']))
        imgs = torch.stack(imgs)
        txts = torch.Tensor(txts)
        target_lengths = torch.Tensor(target_lengths).to(torch.int32)
        # target_lengths = (target_lengths[0],)
        res = {
            'img': imgs,
            'txt': txts,
            'id': ids,
            'target_lengths': target_lengths
        }
        return res


def save_file(res:torch.Tensor, path:str):
    # res = res.cpu().detach().numpy()
    with open(path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)