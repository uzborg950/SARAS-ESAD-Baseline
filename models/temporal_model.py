import torch.nn as nn
import torch
from models.convlstm import ConvLSTM
from models.time_distributed import TimeDistributed5D
import math

class ConvLSTMBlock(nn.Module):
    def __init__(self,input_dim, hidden_dim, num_layers, bias):
        super(ConvLSTMBlock, self).__init__()
        self.num_layers = num_layers
        self.convlstm = self._make_convlstm(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=(3, 3),
                                            num_layers=num_layers,
                                            batch_first=True, bias=bias, return_all_layers=False)

    def forward(self, input, hidden_states):
        #out = torch.unsqueeze(input, 0)
        out = self.convlstm(input, hidden_states=hidden_states)

        #return out[0][0]
        return out[0], out[2]

    def _make_convlstm(self, input_dim, hidden_dim, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True,
                       return_all_layers=True):
        convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers,
                            batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

        for name, param in convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param, a=1)
        return convlstm
class TemporalLayer(nn.Module):
    def __init__(self, inplanes, convlstm_layers, timesteps, use_bias ):
        super(TemporalLayer, self).__init__()
        self.inplanes = inplanes
        self.convlstm = ConvLSTMBlock(inplanes, inplanes, convlstm_layers, use_bias)
        self.td_conv2d =  TimeDistributed5D(make_conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, use_bias=use_bias).cuda(), timesteps)
        self.td_batchnorm = TimeDistributed5D(make_batchnorm(inplanes).cuda(), timesteps)
        self.relu = nn.ReLU(True)



    def forward(self, input_tuple):
        input, hidden_states = input_tuple
        convlstm_out = self.convlstm(input, hidden_states)
        out = self.td_conv2d(convlstm_out[0][0])
        out = self.td_batchnorm(out)
        out = self.relu(out)
        return (out, convlstm_out[1])


def make_batchnorm(inplanes):
        bn=  nn.BatchNorm2d(inplanes)
        bn.weight.data.fill_(1)
        bn.bias.data.zero_()
        return bn
def make_conv2d(inplanes, outplanes, kernel_size, stride=1, padding=1, use_bias=True, init_bg_prob=False, init='he'):
    conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)

    if init == 'he':
        nn.init.kaiming_uniform_(conv.weight, a=1)
    elif init == 'normal':
        nn.init.normal_(conv.weight, mean=0, std=0.01)
    if hasattr(conv.bias, 'data'):
        nn.init.constant_(conv.bias, 0)

    if init_bg_prob and hasattr(conv.bias, 'data'):
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)  # focal mentions this initialization in the paper (page 5)
        nn.init.constant_(conv.bias, bias_value)
    return conv

class TemporalNet(nn.Module):
    def __init__(self, inplanes, outplanes, timesteps, use_bias, init_bg_prob=False, temporal_layers=2, convlstm_layers=1):
        super(TemporalNet, self).__init__()
        self.inplanes = inplanes
        self.temporal_layers = self._make_temporal_net(convlstm_layers, inplanes, temporal_layers, timesteps, use_bias)

        self.td_conv_head = TimeDistributed5D(make_conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, use_bias=use_bias,
                                           init_bg_prob=init_bg_prob, init='normal').cuda(), timesteps)

    def _make_temporal_net(self, convlstm_layers, inplanes, temporal_layers, timesteps, use_bias):
        layers = []
        for _ in range(temporal_layers):
            layers.append(TemporalLayer(inplanes,convlstm_layers, timesteps, use_bias))

        return nn.ModuleList(layers)

    def forward(self, input, hidden_states_layers):
        hidden_states_layers_next = []
        for idx, layer in enumerate(self.temporal_layers):
            out = layer((input, hidden_states_layers[idx] if hidden_states_layers is not None else None))
            hidden_states_layers_next.append(out[1])
            input = out[0]

        out = self.td_conv_head(out[0])
        return out, hidden_states_layers_next

    def _conv1x1(self, in_channel, out_channel, bias):
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)
