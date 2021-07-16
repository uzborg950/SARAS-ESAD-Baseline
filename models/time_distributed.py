import torch.nn as nn
import torch
import copy
''' Converts entire network into Time Distributed layer'''
class TimeDistributed(nn.Module):
    def __init__(self, net, timesteps):
        super(TimeDistributed, self).__init__()
        self.timesteps = timesteps
        self.td_net = net

    def forward(self, input):
        pass

''' For input tensor shape (batch, timesteps, channel, height, width '''
class TimeDistributed5D(TimeDistributed):
    def __init__(self, net, timesteps, fpn=False):
        super(TimeDistributed5D, self).__init__(net, timesteps)
        self.fpn = fpn

    def forward(self, input):
        batch_size, timesteps, channels, height, width = input.shape
        input = input.view(batch_size * timesteps, channels, height, width).contiguous()
        output = self.td_net(input)
        if self.fpn:
            resized_output = []
            for idx, FPN_output in enumerate(output):
                _, channels, height, width = FPN_output.shape
                resized_output.append(FPN_output.view(batch_size, timesteps, channels, height, width))
            return resized_output
        else:
            _, channels, height, width = output.shape
            return output.view(batch_size, timesteps, channels, height, width)


''' For input tensor shape (batch, timesteps, channel, height, width '''


class TimeDistributed3D(TimeDistributed):
    def __init__(self, net, timesteps):
        super(TimeDistributed3D, self).__init__(net, timesteps)

    def forward(self, input):
        batch_size, timesteps, flat = input.shape
        input = input.view(batch_size * timesteps, flat).contiguous()
        output = self.td_net(input)

        _, flat = output.shape
        return output.view(batch_size, timesteps, flat)