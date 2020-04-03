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
    keep_mask = np.zeros([ndets])
    for i_ in range(ndets):
        i = order[i_]
        if suppressed_t[i] != 1:
            keep_mask[i] = 1
            ix1 = boxes[i][0]
            iy1 = boxes[i][1]
            ix2 = boxes[i][2]
            iy2 = boxes[i][3]
            iarea = areas[i]

            for j_ in range(ndets):
                j = order[j_]
                if suppressed_t[j] != 1:
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
    return [a for i, a in enumerate(order) if keep_mask[a]]


def multi_label_nms_kernel(boxes, scores, sorted_indices, iou_thresh):
    # box, score unsorted, indices sorted
    # [99, 4], [99]
    if boxes.shape[0] == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = sorted_indices
    ndets = boxes.shape[0]
    keep_mask = np.zeros([ndets])
    for i_ in range(ndets):
        i = order[i_]
        if scores[i] > 0:
            keep_mask[i] = 1
            ix1 = boxes[i][0]
            iy1 = boxes[i][1]
            ix2 = boxes[i][2]
            iy2 = boxes[i][3]
            iarea = areas[i]

            for j_ in range(ndets):
                j = order[j_]
                if scores[j] > 0:
                    xx1 = max(ix1, boxes[j][0])
                    yy1 = max(iy1, boxes[j][1])
                    xx2 = min(ix2, boxes[j][2])
                    yy2 = min(iy2, boxes[j][3])
                    w = max(0, xx2-xx1)
                    h = max(0, yy2-yy1)

                    inter = w*h
                    ovr = inter / (iarea + areas[j] - inter)
                    if ovr > iou_thresh:
                        scores[j] = 0
    return [a for i, a in enumerate(order) if keep_mask[a]]


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
    sorted_indices = np.argsort(scores)[::-1]
    out_me = multi_label_nms_kernel(rois, scores, sorted_indices, 0.2)
    print('out me: ')
    print(np.array(out_me))

    # # make box sorted by score
    # order = np.argsort(scores)[::-1]
    # print('order: ', order)
    # rois = rois[order]
    # print('rois: ', rois)
    # out_me2 = multi_label2(rois, 0.2)
    # print(out_me2)

    # out_pt = sorted(out_pt)
    print(out_pt - out_me)

if __name__ == "__main__":
    test()


