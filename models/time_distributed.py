import torch.nn as nn
import torch
import copy
''' Converts entire network into Time Distributed layer'''
class TimeDistributed(nn.Module):
    def __init__(self, net, timesteps ):
        super(TimeDistributed, self).__init__()
        self.timesteps = timesteps
        self.td_net = net
        #td_nets = self.create_time_distributed_nets(net, timesteps)

    def create_time_distributed_nets(self, net, timesteps):
        td_nets = []
        td_nets.extend([copy.deepcopy(net) for _ in range(timesteps)])
        return td_nets

    def forward(self, input):
        batch_size, timesteps, channels, height, width = input.shape
        output = torch.tensor([]).cuda()
        for t in range(self.timesteps):
            output_t = self.td_net(input[:,t,:,:,:])
            output_t = torch.unsqueeze(output_t, 1)
            output = torch.cat((output, output_t), 1)
        return output
