"""

Copyright (c) 2019 Gurkirt Singh 
 All Rights Reserved.

"""

import torch.nn as nn
import torch.nn.functional as F
import torch, pdb, time
from modules import box_utils


def sigmoid_focal_loss(preds, labels, num_pos, alpha, gamma):
    '''Args::
        preds: sigmoid activated predictions
        labels: one hot encoded labels
        num_pos: number of positve samples
        alpha: weighting factor to baclence +ve and -ve
        gamma: Exponent factor to baclence easy and hard examples
       Return::
        loss: computed loss and reduced by sum and normlised by num_pos
     '''
    loss = F.binary_cross_entropy(preds, labels, reduction='none')
    alpha_factor = alpha * labels + (1.0 - alpha) * (1.0 - labels)
    pt = preds * labels + (1.0 - preds) * (1.0 - labels)
    focal_weight = alpha_factor * ((1-pt) ** gamma)
    loss = (loss * focal_weight).sum() / num_pos
    return loss


# Credits:: from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/layers/smooth_l1_loss.py
# smooth l1 with beta
def smooth_l1_loss(input, target, beta=1. / 9, reduction='sum'):
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if reduction == 'mean':
        return loss.mean()
    return loss.sum()

# Credits:: https://github.com/amdegroot/ssd.pytorch for multi-box loss
# https://github.com/qfgaohao/pytorch-ssd/blob/master/vision/nn/multibox_loss.py pytorch1.x mulitbox loss
# adopated by Gurkirt Singh
class MultiBoxLoss(nn.Module):
    def __init__(self, positive_threshold, neg_pos_ratio=3):
        """
        
        Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        
        """
        super(MultiBoxLoss, self).__init__()
        self.positive_threshold = positive_threshold
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, gts, counts, anchors):
        
        
        """
        
        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4): predicted locations.
            boxes list of len = batch_size and nx4 arrarys
            anchors: (num_anchors, 4)

        """
        
        num_classes = confidence.size(2)
        gt_locations = []
        labels = []
        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.perf_counter()
            for b in range(len(gts)):
                gt_boxes = gts[b, :counts[b], :4]
                gt_labels = gts[b, :counts[b], 4]
                gt_labels = gt_labels.type(torch.cuda.LongTensor)

                conf, loc = box_utils.match_anchors(gt_boxes, gt_labels, anchors, iou_threshold=self.positive_threshold)

                labels.append(conf)
                gt_locations.append(loc)
            gt_locations = torch.stack(gt_locations, 0)
            labels = torch.stack(labels, 0)
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
        
        # pdb.set_trace()
        pos_mask = labels > 0
        num_pos = max(1.0, float(pos_mask.sum()))

        confidence = confidence[mask, :]
        classification_loss = F.cross_entropy(confidence.reshape(-1, num_classes), labels[mask], reduction='sum') / (num_pos * 4.0)
        
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        
        localisation_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')/(num_pos * 4.0)
        
        return localisation_loss, classification_loss


class YOLOLoss(nn.Module):
    def __init__(self, positive_threshold, negative_threshold):
        """Implement YOLO Loss.
        Basically, combines yolo classification loss
         and Smooth L1 regression loss.
        """
        super(YOLOLoss, self).__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold =negative_threshold
        self.bce_loss = nn.BCELoss().cuda()
        self.pos_weight = 1.0
        self.neg_weight = 0.5


    def forward(self, confidence, predicted_locations, gts, counts, anchors):
        """
        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4): predicted locations.
            boxes list of len = batch_size and nx4 arrarys
            anchors: (num_anchors, 4)

        """
        
        
        confidence = torch.sigmoid(confidence)
        binary_preds = confidence[:,:, 0]
        object_preds = confidence[:,:,1:]
        num_classes = object_preds.size(2)
        N = float(len(gts))
        gt_locations = []
        labels = []
        labels_bin = []
        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.perf_counter()
            for b in range(len(gts)):
                # gt_boxes = gts[b][:,:4]
                # gt_labels = gts[b][:,4]
                gt_boxes = gts[b, :counts[b], :4]
                gt_labels = gts[b, :counts[b], 4]
                gt_labels = gt_labels.type(torch.cuda.LongTensor)

                conf, loc = box_utils.match_anchors_wIgnore(gt_boxes, gt_labels, anchors, 
                                        pos_th=self.positive_threshold, nge_th=self.negative_threshold )

                gt_locations.append(loc)
            
                y_onehot = object_preds.new_zeros(conf.size(0), num_classes+1)
                pos_conf = conf.clone()
                pos_conf[pos_conf<0] = 0 # make ingonre bg
                y_onehot[range(y_onehot.shape[0]), pos_conf] = 1.0
                labels.append(y_onehot[:,1:])
                labels_bin.append(conf)
            
            gt_locations = torch.stack(gt_locations, 0)
            labels = torch.stack(labels, 0)
            labels_bin = torch.stack(labels_bin, 0)

        pos_mask = labels_bin > 0
        num_pos = max(1.0, float(pos_mask.sum()))
        
        predicted_locations = predicted_locations[pos_mask].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask].reshape(-1, 4)
        localisation_loss = smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')/(num_pos * 4.0)
        
        # mask = labels_bin > -1 # Get mask to remove ignore examples
        object_preds = object_preds[pos_mask].reshape(-1,num_classes) # Remove Ignore preds
        labels = labels[pos_mask].reshape(-1, num_classes) # Remove Ignore labels
        # pdb.set_trace()
        classification_loss = F.binary_cross_entropy(object_preds, labels, reduction='sum')/num_pos

        labels_bin = labels_bin.float()
        labels_bin[labels_bin>0] = 1.0
        neg_mask = labels_bin==0
        
        binary_loss_pos = F.binary_cross_entropy(binary_preds[pos_mask], labels_bin[pos_mask], reduction='sum')
        binary_loss_neg = F.binary_cross_entropy(binary_preds[neg_mask], labels_bin[neg_mask], reduction='sum')
        
        binary_loss = (binary_loss_pos*self.pos_weight + binary_loss_neg*self.neg_weight)/num_pos

        # print(classification_loss, binary_loss)
        return localisation_loss, (classification_loss + binary_loss)/2.0


class FocalLoss(nn.Module):
    def __init__(self, positive_threshold, negative_threshold, alpha=0.25, gamma=2.0):
        """Implement YOLO Loss.
        Basically, combines focal classification loss
         and Smooth L1 regression loss.
        """
        super(FocalLoss, self).__init__()
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        # self.bce_loss = nn.BCELoss(reduction='sum').cuda()
        self.alpha = 0.25
        self.gamma = 2.0


    def forward(self, confidence, predicted_locations, gts, counts, anchors):
        
        """
        
        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4): predicted locations.
            boxes list of len = batch_size and nx4 arrarys
            anchors: (num_anchors, 4)

        """

        confidence = torch.sigmoid(confidence)
        binary_preds = confidence[:,:, 0]
        object_preds = confidence[:,:,1:]
        num_classes = object_preds.size(2)
        N = float(len(gts))
        gt_locations = []
        labels = []
        labels_bin = []
        with torch.no_grad():
            # torch.cuda.synchronize()
            # t0 = time.perf_counter()
            for b in range(len(gts)):
                gt_boxes = gts[b, :counts[b], :4]
                gt_labels = gts[b, :counts[b], 4]
                gt_labels = gt_labels.type(torch.cuda.LongTensor)

                conf, loc = box_utils.match_anchors_wIgnore(gt_boxes, gt_labels, anchors, pos_th=self.positive_threshold, nge_th=self.negative_threshold )

                gt_locations.append(loc)
            
                y_onehot = object_preds.new_zeros(conf.size(0), num_classes+1)
                pos_conf = conf.clone()
                pos_conf[pos_conf<0] = 0 # make ingonre bg
                y_onehot[range(y_onehot.shape[0]), pos_conf] = 1.0
                labels.append(y_onehot[:,1:])
                labels_bin.append(conf)
            
            gt_locations = torch.stack(gt_locations, 0)
            labels = torch.stack(labels, 0)
            labels_bin = torch.stack(labels_bin, 0)

        pos_mask = labels_bin > 0
        num_pos = max(1.0, float(pos_mask.sum()))
        
        predicted_locations = predicted_locations[pos_mask].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask].reshape(-1, 4)
        localisation_loss = smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')/(num_pos * 4.0)
        
        mask = labels_bin > -1 # Get mask to remove ignore examples
        object_preds = object_preds[mask].reshape(-1,num_classes) # Remove Ignore preds
        labels = labels[mask].reshape(-1,num_classes) # Remove Ignore labels

        classification_loss = sigmoid_focal_loss(object_preds, labels, num_pos, self.alpha, self.gamma)

        labels_bin[labels_bin>0] = 1
        binary_preds = binary_preds[labels_bin>-1]
        labels_bin = labels_bin[labels_bin>-1]
        binary_loss = sigmoid_focal_loss(binary_preds.float(), labels_bin.float(), num_pos, self.alpha, self.gamma)

        return localisation_loss, (classification_loss + binary_loss)/2.0
