"""

FPN network Classes

Author: Gurkirt Singh
Inspired from https://github.com/kuangliu/pytorch-retinanet and
https://github.com/gurkirt/realtime-action-detection

"""
import math

import torch
import torch.nn as nn

from models.backbone_models import backbone_models
from models.temporal_model import TemporalNet
from modules.anchor_box_retinanet import anchorBox
from modules.box_utils import decode
from modules.detection_loss import MultiBoxLoss, YOLOLoss, FocalLoss


class RetinaNet(nn.Module):
    """Feature Pyramid Network Architecture
    The network is composed of a backbone FPN network followed by the
    added Head conv layers.  
    Each head layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions

    See: 
    RetinaNet: https://arxiv.org/pdf/1708.02002.pdf for more details.
    FPN: https://arxiv.org/pdf/1612.03144.pdf

    Args:
        backbone Network:
        Program Argument Namespace

    """

    def __init__(self, backbone, args):
        super(RetinaNet, self).__init__()

        self.num_classes = args.num_classes
        # TODO: implement __call__ in 

        self.anchors = anchorBox()
        self.ar = self.anchors.ar
        args.ar = self.ar
        self.use_bias = args.use_bias
        self.head_size = args.head_size
        self.backbone_net = backbone
        self.shared_heads = args.shared_heads
        self.num_head_layers = args.num_head_layers
        self.append_temporal_net = args.append_temporal_net
        self.predictor_layers = args.predictor_layers
        self.starting_prediction_layer = args.starting_prediction_layer
        self.convlstm_layers = args.convlstm_layers
        self.temporal_net_layers = args.temporal_net_layers

        assert self.shared_heads < self.num_head_layers, 'number of shared layers should be less than head layers h:' + str(
            self.num_head_layers) + ' sh:' + str(self.shared_heads)

        if self.shared_heads > 0:
            self.features_layers = self.make_features(self.shared_heads)

        if not self.append_temporal_net:
            self.reg_heads = self.make_head(self.ar * 4,
                                            self.num_head_layers - self.shared_heads)  # box subnet. Figure 3 retinanet paper: W x H x 4A (A=num of anchors)
            self.cls_heads = self.make_head(self.ar * self.num_classes,
                                            self.num_head_layers - self.shared_heads)  # class subnet. W x H x KA (K=number of action classes)

        else:
            self.reg_temporal = TemporalNet(self.head_size, self.ar * 4, use_bias=self.use_bias, init_bg_prob=False,
                                            temporal_layers=self.temporal_net_layers,
                                            convlstm_layers=self.convlstm_layers)
            self.cls_temporal = TemporalNet(self.head_size, self.ar * self.num_classes, use_bias=self.use_bias,
                                            init_bg_prob=True, temporal_layers=self.temporal_net_layers,
                                            convlstm_layers=self.convlstm_layers)

        if args.loss_type != 'mbox' and not self.append_temporal_net:
            self.prior_prob = 0.01
            bias_value = -math.log(
                (1 - self.prior_prob) / self.prior_prob)  # focal mentions this initialization in the paper (page 5)
            nn.init.constant_(self.cls_heads[-1].bias, bias_value)
        if not hasattr(args, 'eval_iters'):  # eval_iters only in test case
            if args.loss_type == 'mbox':
                self.criterion = MultiBoxLoss(args.positive_threshold)
            elif args.loss_type == 'yolo':
                self.criterion = YOLOLoss(args.positive_threshold, args.negative_threshold)
            elif args.loss_type == 'focal':
                self.criterion = FocalLoss(args.positive_threshold, args.negative_threshold)
            else:
                error('Define correct loss type')

    def forward(self, images, gts=None, counts=None, get_features=False):
        sources = self.backbone_net(images)
        features = list()
        # pdb.set_trace()
        if self.shared_heads > 0:
            for x in sources:
                features.append(self.features_layers(x))
        else:
            features = sources  # size 5 Tensors tuple (p3 to p7). P3 is [2,256,28,48]

        grid_sizes = [feature_map.shape[-2:] for feature_map in
                      features]  # Gets grid size, skips num of planes/channels
        anchor_boxes = self.anchors(grid_sizes) #Contains anchor boxes at each grid size for each pixel in the image (increasing strides for decreasing grid sizes)

        loc = list()
        conf = list()

        for x in features:
            if not self.append_temporal_net:
                reg_out = self.reg_heads(x).permute(0, 2, 3, 1).contiguous()  # 2,25,45,36 (same as P3 w,h)
                cls_out = self.cls_heads(x).permute(0, 2, 3, 1).contiguous()
            else:
                reg_out = self.reg_temporal(x).permute(0, 2, 3, 1).contiguous()  # 2,25,45,36 (same as P3 w,h)
                cls_out = self.cls_temporal(x).permute(0, 2, 3, 1).contiguous()

            loc.append(reg_out)  # batch size, height, width, channels=4*anchors (4*9)
            conf.append(cls_out)  # batch size, height, width, channels=classes*anchors (22*9)

        loc = torch.cat([o.reshape(o.size(0), -1) for o in loc],
                        1)  # For box subnet output of each layer, flattens it while preserving batch size
        conf = torch.cat([o.reshape(o.size(0), -1) for o in conf], 1)

        flat_loc = loc.view(loc.size(0), -1, 4)  # batch size ,x,4 ->  For each pixel of the image and for all anchors at all grid sizes, give coords of box
        flat_conf = conf.view(conf.size(0), -1, self.num_classes)  # batch size ,x,22 -> For each pixel of the image and for all anchors at all grid sizes, give confidence of action
        # pdb.set_trace()

        if get_features:  # testing mode with feature return
            return torch.stack([decode(flat_loc[b], anchor_boxes) for b in range(flat_loc.shape[0])],
                               0), flat_conf, features
        elif gts is not None:  # training mode
            return self.criterion(flat_conf, flat_loc, gts, counts, anchor_boxes, images)
        else:  # otherwise testing mode
            return torch.stack([decode(flat_loc[b], anchor_boxes) for b in range(flat_loc.shape[0])], 0), flat_conf

    def make_features(self, shared_heads):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        for _ in range(shared_heads):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1, padding=1, bias=use_bias))
            layers.append(nn.ReLU(True))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

    def make_head(self, out_planes, num_non_shared_heads):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size  # 256
        for _ in range(num_non_shared_heads):
            layers.append(nn.Conv2d(head_size, head_size, kernel_size=3, stride=1, padding=1, bias=use_bias))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(head_size, out_planes, kernel_size=3, stride=1, padding=1))  # actual predictor head

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers


def build_retinanet_shared_heads(args):
    # print('basenet', args.basenet)
    backbone = backbone_models(args.basenet, args.model_dir, args.use_bias, args)
    # print('backbone model::', backbone)
    return RetinaNet(backbone, args)
