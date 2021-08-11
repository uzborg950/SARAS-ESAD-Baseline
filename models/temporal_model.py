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

    def forward(self, input, reset_hidden, detach_state):
        #out = torch.unsqueeze(input, 0)
        out = self.convlstm(input, reset_hidden= reset_hidden, detach_state=detach_state)
        #out = torch.squeeze(out[0][0], 0)
        return out[0][0]

    def _make_convlstm(self, input_dim, hidden_dim, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True,
                       return_all_layers=True):
        convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers,
                            batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

        for name, param in convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        return convlstm
class TemporalLayer(nn.Module):
    def __init__(self, inplanes, convlstm_layers, timesteps, use_bias ):
        super(TemporalLayer, self).__init__()
        self.inplanes = inplanes
        self.convlstm = ConvLSTMBlock(inplanes, inplanes, convlstm_layers, use_bias)
        self.td_conv2d =  TimeDistributed5D(make_conv2d(inplanes, inplanes, kernel_size=3, stride=1, padding=1, use_bias=use_bias).cuda(), timesteps)
        self.td_batchnorm = TimeDistributed5D(nn.BatchNorm2d(inplanes).cuda(), timesteps)
        self.relu = nn.ReLU(True)
    def forward(self, input_tuple):
        input, reset_hidden, detach_state = input_tuple
        out = self.convlstm(input, reset_hidden, detach_state)
        out = self.td_conv2d(out)
        out = self.td_batchnorm(out)
        out = self.relu(out)
        return out, reset_hidden, detach_state



def make_conv2d(inplanes, outplanes, kernel_size, stride=1, padding=1, use_bias=True, init_bg_prob=False):
    conv = nn.Conv2d(inplanes, outplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=use_bias)

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
        self.temporal_net = self._make_temporal_net(convlstm_layers, inplanes, temporal_layers, timesteps, use_bias)

        self.td_conv_head = TimeDistributed5D(make_conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=1, use_bias=use_bias,
                                           init_bg_prob=init_bg_prob).cuda(), timesteps)

    def _make_temporal_net(self, convlstm_layers, inplanes, temporal_layers, timesteps, use_bias):
        layers = []
        for _ in range(temporal_layers):
            layers.append(TemporalLayer(inplanes,convlstm_layers, timesteps, use_bias))
        net = nn.Sequential(*layers)
        return net

    def forward(self, input, temporal_flags):
        out = self.temporal_net((input, temporal_flags[0], temporal_flags[1]))
        out = self.td_conv_head(out[0])
        return out

    def _conv1x1(self, in_channel, out_channel, bias):
        return nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=bias)

    def make_fc_layers(self, num_fc_layers, inplanes, output_size, activation=False):
        layers = []
        for _ in range(num_fc_layers):
            layers.append(nn.Linear(inplanes, output_size))
            if activation:
                layers.append(nn.LeakyReLU(inplace=True))

        layers = nn.Sequential(*layers)
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                for name, param in m.named_parameters():
                    if 'bias' in name:
                        nn.init.constant(param, 0.0)
                    elif 'weight' in name:
                        nn.init.xavier_normal(param)
        return layers

    def _make_convlstm(self, input_dim, hidden_dim, kernel_size=(3, 3), num_layers=1, batch_first=True, bias=True,
                       return_all_layers=True):
        convlstm = ConvLSTM(input_dim=input_dim, hidden_dim=hidden_dim, kernel_size=kernel_size, num_layers=num_layers,
                            batch_first=batch_first, bias=bias, return_all_layers=return_all_layers)

        for name, param in convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)
        return convlstm

    def make_lstm(self, in_planes):
        # for layer_num in range(self.predictor_layers):
        # Predictor layers can be used to create one lstm per predictor layer.

        # LSTMs only need input of planes, not the complete flattened dimension unlike FC layers
        # The LSTM hidden layer/output dimension is kept same as input as it is to be later used while finding anchor box loss
        # LSTMs will be shared (Because predictor heads in retinanet were shared. We also can't afford the added computational costs)
        lstm = nn.LSTM(in_planes, in_planes, batch_first=True)

        # Initialize lstm with xavier initailization
        for name, param in lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal(param)

        return lstm
