import io
import numpy as np

from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
import torch


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
        a = x[a]
        print(a.shape)
        return a


torch_model = MB()
x = torch.randn(1, 256, 224)
torch_out = torch_model(x)

# Export the model
torch.onnx.export(torch_model,               # model being run
                  x,
                  "a.onnx",
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True)
print('Done!')
