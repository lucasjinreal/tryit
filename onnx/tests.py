import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch
import itertools
import operator
import torch.nn.functional as F
from utils import triu_onnx

# https://github.com/onnx/onnx-tensorrt/issues/506


def gather(input, dim, index):
    indices = [torch.arange(size, device=index.device) for size in index.shape]
    indices = list(torch.meshgrid(*indices))
    indices[dim] = index
    sizes = list(
        reversed(list(itertools.accumulate(reversed(input.shape), operator.mul))))
    index = sum((index * size for index,
                 size in zip(indices, sizes[1:] + [1])))
    output = input.flatten()[index]
    return output


class M(nn.Module):

    def forward(self, x):
        x_pos = torch.arange(-1, 1, 2/x.shape[-2])
        y_pos = torch.arange(-1, 1, 2/x.shape[-1])
        return torch.meshgrid([x_pos, y_pos])


class MA(nn.Module):

    def forward(self, x):
        x_pos = torch.tensor([4, 5, 6, 7])
        p_pos = torch.tensor([2, 5, 2, 7])
        a = x_pos * p_pos
        b = torch.randn([1, 64, 14336])
        b = torch.matmul(x, b)
        return b


class MB(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # prob = x
        # idxs = torch.argmax(prob, dim=-1)
        # idxs = idxs[:,:40]
        # print(idxs.shape)
        # scores = prob.squeeze(0)[idxs]
        # print(scores.shape)
        # scores = torch.gather(prob, -1, idxs)

        a = torch.randint(0, 224, [1, 256, 50])
        # a = torch.gather(x, dim=2, index=a)
        a = gather(x, dim=2, index=a)
        print(a.shape)
        return a


class MC(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x = torch.argmax(x)
        _, keep = torch.sort(x, descending=True)
        a = x[keep]
        print(a.shape)
        return a


class MD(nn.Module):

    def __init__(self):
        super().__init__()

        self.embed = nn.Conv2d(256, 100, 1, 1)

    def forward(self, x):
        # x: 100, 256, 1, 1
        # x = torch.argmax(x)
        # a = x.triu(diagonal=1)
        # b = triu_onnx(x, diagonal=1)
        # print(b)
        # print(a == b)

        seg_preds = torch.randn([1, 256, 120, 64])
        self.embed.weight = torch.nn.Parameter(x)
        a = self.embed(seg_preds)
        # b = F.conv2d(seg_preds, x, stride=1).squeeze(0).sigmoid()
        print(a.shape)  # 1,100,120,64
        return a


class ME(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        idx = torch.randn([17, 1])
        width = idx.shape[0]
        if isinstance(width, torch.Tensor):
            width = width.to(torch.float)
        print(type(width))
        # preds = torch.cat([idx, idx], dim=1).to(torch.float)
        preds = torch.cat([idx, idx], dim=1).to(torch.float)

        preds[:, 0] = (preds[:, 0]) % width
        preds[:, 1] = torch.floor((preds[:, 1]) / width)
        return preds


class MF(nn.Module):

    def __init__(self):
        super().__init__()
        # for test if torch.cat([bool, bool]) can convert

    def forward(self, x):
        x = x.to(torch.float)
        print(x.shape)
        mask = x > 0.2
        print(mask)
        preds = torch.cat([mask, mask], dim=1)
        return preds


class MG(nn.Module):

    def __init__(self):
        super().__init__()
        # for test if torch.cat([bool, bool]) can convert

    def forward(self, x, b):
        # x, b = x
        preds = F.conv2d(x, b,
                             stride=1)
        # preds = preds.to(torch.float)
        # preds = preds.sigmoid().float()
        # seg_masks = preds > torch.tensor(0.03, dtype=torch.float)
        # return seg_masks
        return preds


torch_model = MG()
x = torch.randn([1, 4, 24, 24])
b = torch.randn([8, 4, 3, 3])
torch_out = torch_model(x, b)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  (x, b),
                  "a.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,
                  verbose=True)
print('Done!')
