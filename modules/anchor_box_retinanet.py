import torch
from math import sqrt as sqrt
from itertools import product as product
import numpy as np

# from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
class BufferList(torch.nn.Module):
    """
    
    Similar to nn.ParameterList, but for buffers
    
    """

    def __init__(self, buffers=None):
        super(BufferList, self).__init__()
        if buffers is not None:
            self.extend(buffers)

    def extend(self, buffers):
        offset = len(self)
        for i, buffer in enumerate(buffers):
            self.register_buffer(str(offset + i), buffer)
        return self

    def __len__(self):
        return len(self._buffers)

    def __iter__(self):
        return iter(self._buffers.values())


class anchorBox(torch.nn.Module):
    """Compute anchorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, anchor_type = 'pdf9', sizes = [32, 64, 128, 256, 512],
                        ratios = np.asarray([0.5, 1 / 1., 2.0]),
                        strides = [8, 16, 32, 64, 128],
                        scales = np.array([1, 1.25992, 1.58740])):

        super(anchorBox, self).__init__()
        self.sizes = sizes
        self.ratios = ratios
        self.scales = scales
        self.strides = strides
        if anchor_type != 'pdf9':
            self.scales = np.array([2 ** 0,])
        self.ar = len(self.scales)*len(self.ratios) #Shouldn't this be len(scales) * len(ratios)?
        self.cell_anchors = BufferList(self._get_cell_anchors())
        
    def _get_cell_anchors(self):
        anchors = []
        for s1 in self.sizes: #For each size, we create anchor matrix of size [9,4] [scale*ratio, coords]
            p_anchors = np.asarray(self._gen_generate_anchors_on_one_level(s1))
            p_anchors = torch.FloatTensor(p_anchors).cuda()
            anchors.append(p_anchors)

        return anchors
    
    # modified from https://github.com/fizyr/keras-retinanet/blob/master/keras_retinanet/utils/anchors.py
    # Copyright 2017-2018 Fizyr (https://fizyr.com)
    def _gen_generate_anchors_on_one_level(self, base_size=32):
        
        """
        Generate anchor (reference) windows by enumerating aspect ratios X
        scales w.r.t. a reference window.
        
        """

        num_anchors = len(self.ratios) * len(self.scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = base_size * np.tile(self.scales, (2, len(self.ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3] #2nd and 3rd index contain same elements at each row [32,32],[40,40].. * 9

        anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales))
        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
        return anchors

    # forward from https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/modeling/rpn/anchor_generator.py
    def forward(self, grid_sizes):
        
        anchors = []
        for size, stride, base_anchors in zip(grid_sizes, self.strides, self.cell_anchors): #For each grid size, we have to resize our base anchors and create new anchors for each stride
            grid_height, grid_width = size
            device = base_anchors.device
            shifts_x = torch.arange(0, grid_width, dtype=torch.float32).cuda()
            shifts_y = torch.arange(0, grid_height, dtype=torch.float32).cuda() 
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x) 
            shift_x = (shift_x.reshape(-1) + 0.5) * stride #Notice in each iteration, with smaller grid sizes, strides are bigger
            shift_y = (shift_y.reshape(-1) + 0.5) * stride
            shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1) #[grid height x width, 4)
            anchors.append( (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4)).reshape(-1, 4) ) #Resize base anchors based on grid size for each pixel coordinate of image 90450 / (width * height) = 9

        return torch.cat(anchors, 0) #The smaller grid sizes have larger strides and fewer anchors
        
