"""

Copyright (c) 2019 Gurkirt Singh 
 All Rights Reserved.

"""

import json
import csv
import torch
import pdb, time
import torch.utils.data as data
import pickle
from .transforms import get_image_list_resized
import torch.nn.functional as F
import numpy as np
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image, ImageDraw
import glob
import pdb

def read_file(path, full_test):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except:
        lines = None

    if lines is None and full_test:
        # in case we are testing and we don't have labels
        # but we need to return at least one label per image
        # we fake it like belwo
        return [[0.25, 0.25, 0.75, 0.75, -1]]

    if not lines: 
        # during training or validation we return None as labels so we can skip the image
        return None
    
    
    # during training or validation we return the labels
    lines = [line.split(' ') for line in lines if len(line)>0]

    out_data = []
    for line in lines:
        line_entries = [float(entry) for entry in line]
        line_entries = [line_entries[1], line_entries[2], line_entries[3], line_entries[4], line_entries[0]]
        out_data.append(line_entries)
    
    return out_data

def read_labels(image_files, full_test):
    labels=[]
    
    for img_path in image_files:
        label_file = img_path.replace('.jpg', '.txt')
        label= read_file(label_file, full_test)
        if label is not None:
            labels.append([img_path, label])
    
    return labels


def read_sets(path, input_sets=['train/set1','train/set2'], full_test=False):
    
    all_files=[]
    for set_name in input_sets:
        set_path= path + set_name
        image_files= glob.glob(set_path+'/*.jpg')
        all_files.extend(image_files)
        
    labels= read_labels(all_files, full_test)
    print('length of labels', len(labels))
    return(labels)
    
            
def make_object_lists(rootpath, input_sets=['train/set1','train/set2'], full_test=False):
    '''

    input_sets has be a list of set needs tobe read : 
    e.g. 'train/set1','train/set2', 'val/obj', or 'test/obj'
    

    '''

    with open(rootpath+'train/obj.names', 'r') as fil:
        cls_list= fil.read().split('\n')

    cls_list = [name for name in cls_list if len(name)>0]
    
    final_labels= read_sets(rootpath, input_sets, full_test)
        
    return(cls_list, final_labels)


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class DetectionDataset(data.Dataset):
    """Detection Dataset class for pytorch dataloader"""

    def __init__(self, root, train=False, input_sets=['train/set1','train/set2'], transform=None, anno_transform=None, full_test=False):

        self.train = train
        self.root= root
        self.input_sets = input_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        self.classes, self.ids = make_object_lists(self.root, input_sets=input_sets, full_test=full_test)
        self.print_str= ''
        self.max_targets = 20
        
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        annot_info = self.ids[index]
        
        img_path = annot_info[0]
        bbox_info = np.array(annot_info[1])
        labels= bbox_info[:,4]

        x1= bbox_info[:,0] - bbox_info[:,2]/2
        y1= bbox_info[:,1] - bbox_info[:,3]/2
        x2= bbox_info[:,0] + bbox_info[:,2]/2
        y2= bbox_info[:,1] + bbox_info[:,3]/2

        boxes= np.hstack([np.expand_dims(x1, axis=1), np.expand_dims(y1, axis=1),
                          np.expand_dims(x2, axis=1), np.expand_dims(y2, axis=1)])
        # Box coordinates should be in ranges of [0,1]

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        ## Horizontal flip is turned off because some catories are sentive to the flip.
        # if self.train and np.random.random() < 0.5:
        #     img = img.transpose(Image.FLIP_LEFT_RIGHT)
        #     w = boxes[:, 2] - boxes[:, 0]
        #     boxes[:, 0] = 1 - boxes[:, 2] # boxes should be in x1 y1 x2 y2 [0,1] format 
        #     boxes[:, 2] = boxes[:, 0] + w # boxes should be in x1 y1 x2 y2 [0,1] format
        
        if self.transform is not None:
            img = self.transform(img)

        _, height, width = img.shape

        wh = [width, height, orig_w, orig_h]
        # print(wh)
        boxes[:, 0] *= width # width x1
        boxes[:, 2] *= width # width x2
        boxes[:, 1] *= height # height y1
        boxes[:, 3] *= height # height y2

        targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, targets, index, wh
        

def custum_collate(batch):
    targets = []
    images = []
    image_ids = []
    whs = []
    # fno = []
    # rgb_images, flow_images, aug_bxsl, prior_labels, prior_gt_locations, num_mt, index
    
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2])
        whs.append(sample[3])
    
    counts = []
    max_len = -1
    for target in targets:
        max_len = max(max_len, target.shape[0])
        counts.append(target.shape[0])
    new_targets = torch.zeros(len(targets), max_len, targets[0].shape[1])
    cc = 0
    for target in targets:
        new_targets[cc,:target.shape[0]] = target
        cc += 1
    images_ = get_image_list_resized(images)
    cts = torch.LongTensor(counts)
    # print(images_.shape)
    return images_, new_targets, cts, image_ids, whs

# if __name__== '__main__':
# #    from torchvision import transforms
#     dataset= DetectionDataset(root= folder, image_sets= 'train')
#     dataset_val= DetectionDataset(root= folder, image_sets= 'val')
#     print('train',len(dataset))
#     print('val',len(dataset_val))
