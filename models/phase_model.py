import math
import torch
import torch.nn as nn

import modules.image_utils as img_utils
import modules.resnet_utils as resnet_utils


class PhaseHead(nn.Module):
    def __init__(self, input_dim, inplanes, num_phases):
        super(PhaseHead, self).__init__()
        self.height = input_dim[1]
        self.width = input_dim[0]
        self.inplanes= inplanes
        self.fc1 = nn.Linear(self.height * self.width * self.inplanes, 128) #Multiply channels too
        self.relu = nn.ReLU(True)

        self.fc2 = nn.Linear(128, num_phases)

    def forward(self, input):
        out = self.fc1(input)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class PhaseNet(nn.Module):
    def __init__(self, temporal_cls_backbone, inplanes, args):
        super(PhaseNet, self).__init__()
        self.num_phases = args.num_phases
        self.temporal_cls_backbone = temporal_cls_backbone
        self.phase_heads = []
        self.feature_layers = args.predictor_layers
        for feature_layer in self.feature_layers:
            dim = img_utils.get_size((args.original_width, args.original_height), args.min_size, args.max_size)
            dim = resnet_utils.get_dimensions(dim, feature_layer)
            phase_head = PhaseHead(dim,inplanes, self.num_phases).cuda()
            self.phase_heads.append(phase_head)
        self.fc = nn.Linear(len(self.feature_layers) * self.num_phases, self.num_phases)

    def forward(self, feature_layer_inputs):
        phase_out = []
        for layer_num, feature in enumerate(feature_layer_inputs):
            cls_out = self.temporal_cls_backbone(feature)
            cls_out = torch.flatten(cls_out, start_dim=1)
            phase_out.append(self.phase_heads[layer_num](cls_out))

        phase = torch.cat([o.reshape(o.size(0), -1) for o in phase_out], 1)
        out_phase = self.fc(phase)
        return out_phase


