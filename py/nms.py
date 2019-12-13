import numpy as np
from torchvision.ops import nms
from alfred.dl.torch.common import device
import torch
from torch import nn


torch.manual_seed(1024)
np.random.seed(1024)


class TinyModel(nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()
        self.expander = nn.Conv1d(100, 1, 1, 1)
        self.atom = torch.tensor([[0.22], [0.14], [0.1], [0.4]]).to(device)

    def forward(self, x: torch.Tensor):
        rois = x['rois']
        score = x['scores']
        x = nms(rois, score, 0.2)
        return x


def multi_label_nms(boxes, scores, iou_thresh):
    # [99, 4], [99]
    if boxes.shape[0] == 0:
        return []
    
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(scores)[::-1]

    ndets = boxes.shape[0]
    suppressed_t = np.zeros([ndets])
    # keep_t = np.zeros([ndets])
    keep_t = np.zeros([ndets])

    num_keep = 0
    for i_ in range(ndets):
        i = order[i_]
        if suppressed_t[i] == 1:
            continue
        keep_t[num_keep] = i
        num_keep += 1

        ix1 = boxes[i][0]
        iy1 = boxes[i][1]
        ix2 = boxes[i][2]
        iy2 = boxes[i][3]
        iarea = areas[i]

        for j_ in range(ndets):
            j = order[j_]
            if suppressed_t[j]  == 1:
                continue
            xx1 = max(ix1, boxes[j][0])
            yy1 = max(iy1, boxes[j][1])
            xx2 = min(ix2, boxes[j][2])
            yy2 = min(iy2, boxes[j][3])

            w = max(0, xx2-xx1)
            h = max(0, yy2-yy1)

            inter = w*h
            ovr = inter / (iarea + areas[j] - inter)
            if ovr > iou_thresh:
                suppressed_t[j] = 1
    return keep_t[:num_keep]


def multi_label2(boxes, iou_thresh):
    # assume boxes already sorted with score
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    n_dets = boxes.shape[0]
    mask = np.zeros([n_dets])
    keep = np.zeros([n_dets])
    num_keep = 0
    for i in range(n_dets):
        if mask[i] == 1:
            continue
        keep[num_keep] = i
        num_keep += 1
        iarea =areas[i]
        ix1 = boxes[i][0]
        iy1 = boxes[i][1]
        ix2 = boxes[i][2]
        iy2 = boxes[i][3]

        for j in range(n_dets):
            xx1 = max(ix1, boxes[j][0])
            yy1 = max(iy1, boxes[j][1])
            xx2 = min(ix2, boxes[j][2])
            yy2 = min(iy2, boxes[j][3])
            w = max(0, xx2-xx1)
            h = max(0, yy2-yy1)
            jarea = areas[j]
            inter = w*h
            ovr = inter / (iarea + jarea - inter)
            if ovr > iou_thresh:
                mask[j] = 1
    return keep

def test():
    model = TinyModel().to(device)
    rois = torch.rand(99, 4).to(device)
    scores = torch.rand(99).to(device)
    sample_input = {
        'rois': rois,
        'scores': scores
    }
    model.eval()
    
    with torch.no_grad():
        out_pt = model(sample_input)
        out_pt = out_pt.cpu().numpy()
    print('pytorch output: {}'.format('-'*80))
    print(out_pt)
    print(out_pt.shape)

    rois = rois.cpu().numpy()
    scores = scores.cpu().numpy()
    out_me = multi_label_nms(rois, scores, 0.2)
    print('out me: ', out_me)

    # make box sorted by score
    order = np.argsort(scores)[::-1]
    print('order: ', order)
    rois = rois[order]
    print('rois: ', rois)
    out_me2 = multi_label2(rois, 0.2)
    print(out_me2)

    # out_pt = sorted(out_pt)
    print(out_pt - out_me)

if __name__ == "__main__":
    test()


