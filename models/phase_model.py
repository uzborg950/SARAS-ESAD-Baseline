import torch
import torch.nn as nn
import modules.image_utils as img_utils
import modules.resnet_utils as resnet_utils


class PhaseNet(nn.Module):
    def __init__(self, cls_inplanes, args):
        super(PhaseNet, self).__init__()
        self.num_classes = args.num_classes
        self.num_phases = args.num_phases
        self.fpn_output_layers = args.predictor_layers

        input_dim = img_utils.get_size((args.original_width, args.original_height), args.min_size, args.max_size)
        self.phase_tail = self.make_phase_tail(input_dim, cls_inplanes)
        self.phase_head = nn.Linear(self.num_classes, self.num_phases)
        self.bn = nn.BatchNorm1d(self.num_classes)
        self.relu = nn.ReLU(inplace=True)

        self._init_hidden_units(self.phase_tail, init='he')
        self._init_hidden_units(self.phase_head, init='normal')

    def _init_hidden_units(self, hidden_units, init):
        nn.init.constant(hidden_units.bias, 0.0)

        if init == 'he':
            nn.init.kaiming_uniform_(hidden_units.weight, a=1)
        elif init == 'normal':
            nn.init.normal_(hidden_units.weight, mean=0, std=0.01)
    def make_phase_tail(self, input_dim, cls_inplanes):
        flat_cls_output_dim = 0
        for fpn_output_layer in self.fpn_output_layers:
            fpn_output_dim = resnet_utils.get_dimensions(input_dim, fpn_output_layer)
            flattened_dim = fpn_output_dim[0] * fpn_output_dim[1] * cls_inplanes
            flat_cls_output_dim += flattened_dim
        return nn.Linear(flat_cls_output_dim, self.num_classes)

    def forward(self, flat_cls_output):
        out = self.phase_tail(torch.flatten(flat_cls_output, start_dim=1))
        out = self.bn(out) #To avoid extremely large values, otherwise relu never gets a chance to act non-linear and zero out
        out = self.relu(out)
        out_phase = self.phase_head(out)
        return out_phase
