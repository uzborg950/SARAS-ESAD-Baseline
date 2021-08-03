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
import math
def read_file(path, full_test, include_phase):
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
    except:
        lines = None

    if not lines and full_test:
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
        if include_phase and len(line) == 1:
            line_entries = [float(line[0]), -1.0, -1.0, -1.0, -1.0]
        elif len(line) == 5:
            line_entries = [float(entry) for entry in line]
        else:
            continue

        line_entries = [line_entries[1], line_entries[2], line_entries[3], line_entries[4], line_entries[0]]
        out_data.append(line_entries)
    
    return out_data

def read_labels(image_files, full_test, include_phase, batch):
    labels=[]
    input_set_sizes = []
    count = 0
    for img_path in image_files:
        if img_path == -1:
            if batch is not None: #put dummy data to fill out remaining batch
                for _ in range(batch - int(math.modf(count/batch)[0] * batch)):
                    labels.append(-1)
                count = count + batch - int(math.modf(count/batch)[0] * batch)
            input_set_sizes.append(count)
            count = 0
            continue
        label_file = img_path.replace('.jpg', '.txt')
        label= read_file(label_file, full_test, include_phase=include_phase)
        if label is not None:
            labels.append([img_path, label])
            count +=1
    
    return labels, input_set_sizes

def frame_number_sorter(item):
    return int(item.split("_")[-1].split(".")[0])
def read_sets(path, input_sets=['train/set1','train/set2'], full_test=False, include_phase=False, batch=None):
    
    all_files=[]
    for set_name in input_sets:
        set_path= path + set_name
        image_files= sorted(glob.glob(set_path+'/*.jpg'), key=frame_number_sorter)
        all_files.extend(image_files)
        all_files.append(-1)
        
    labels, input_set_sizes = read_labels(all_files, full_test, include_phase=include_phase, batch=batch)

    print('length of labels', len(labels))
    return(labels, input_set_sizes)
    
            
def make_object_lists(rootpath, input_sets=['train/set1','train/set2'], full_test=False, include_phase= False, batch= None):
    '''

    input_sets has be a list of set needs tobe read : 
    e.g. 'train/set1','train/set2', 'val/obj', or 'test/obj'
    

    '''

    with open(rootpath+'train/obj.names', 'r') as fil:
        cls_list= fil.read().split('\n')

    cls_list = [name for name in cls_list if len(name)>0]
    
    final_labels, input_set_sizes = read_sets(rootpath, input_sets, full_test, include_phase=include_phase, batch= batch)
        
    return(cls_list, final_labels, input_set_sizes)


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class DetectionDataset(data.Dataset):
    """Detection Dataset class for pytorch dataloader"""

    def __init__(self, root, train=False, input_sets=['train/set1','train/set2'], transform=None, anno_transform=None, full_test=False, include_phase=False, batch=None):
        self.include_phase = include_phase
        self.train = train
        self.root= root
        self.input_sets = input_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        self.classes, self.ids, self.input_set_sizes = make_object_lists(self.root, input_sets=input_sets, full_test=full_test, include_phase=self.include_phase, batch=batch)
        self.print_str= ''
        self.max_targets = 20
        
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        annot_info = self.ids[index]

        if annot_info == -1: #Dummy batch-filler data
            return torch.tensor(-1, dtype=torch.float32), -1, -1, -1

        img_path = annot_info[0]
        bbox_info = np.array(annot_info[1]) #shape: (G, 5) , G is the number of actions in frame. 5 is box coords (4) + action class(1)
        phase = -1
        if self.include_phase:
            phase = bbox_info[-1,:]
            bbox_info = bbox_info[:-1,:]

        labels= bbox_info[:,4] #Gets action ids in the frame

        x1= bbox_info[:,0] - bbox_info[:,2]/2 #Upper left x 
        y1= bbox_info[:,1] - bbox_info[:,3]/2 #Upper left y 
        x2= bbox_info[:,0] + bbox_info[:,2]/2 #Lower right x 
        y2= bbox_info[:,1] + bbox_info[:,3]/2 #Lower right y 

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
        boxes[:, 0] *= width # width x1 (GT coords are normalized and need to be scaled by image size)
        boxes[:, 2] *= width # width x2
        boxes[:, 1] *= height # height y1
        boxes[:, 3] *= height # height y2

        targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        if(self.include_phase):
            targets = np.vstack((targets, np.expand_dims(phase, axis=0)))
        return img, targets, index, wh


def get_non_empty_sample(batch):
    for sample in batch:
        if not torch.equal(sample[0], torch.tensor(-1, dtype=torch.float32)):
            return sample[0], torch.FloatTensor(sample[1]), sample[2], sample[3]


def custom_collate(batch, timesteps):  # batch size 16
    targets = []
    images = []
    image_ids = []
    whs = []
    # fno = []
    # rgb_images, flow_images, aug_bxsl, prior_labels, prior_gt_locations, num_mt, index

    for sample in batch:
        if torch.equal(sample[0], torch.tensor(-1, dtype=torch.float32)):
            if not images: #Hack for if the first image is -1 (this can only come up when shuffle true)
                image, target, id, whsi = get_non_empty_sample(batch)
                images.append(image.clone())
                targets.append(target.clone())
                image_ids.append(id)
                whs.append(whsi)
                continue
            images.append(images[-1].clone())
            targets.append(targets[-1].clone())
            image_ids.append(image_ids[-1])
            whs.append(whs[-1])
            continue
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
        image_ids.append(sample[2]) #Index while iterating total list of samples (randomized iteration)
        whs.append(sample[3]) #[width (1067), height (600), orig_w (1920), orig_h (1080)] (same for all tensors)
    for i in range(timesteps - len(batch)): #Repeat the last image to fill batch (consistency required for model expecting sequences (timesteps) of frames)
        images.append(images[-1].clone())
        targets.append(targets[-1].clone())
        image_ids.append(image_ids[-1])
        whs.append(whs[-1])
    counts = []
    max_len = -1
    for target in targets:
        max_len = max(max_len, target.shape[0]) #Max actions a sample could have in the whole dataset
        counts.append(target.shape[0]) #Stores number of action classes in each sample
    new_targets = torch.zeros(len(targets), max_len, targets[0].shape[1]) #shape (16 (targets/batch size),2 (max actions in a sample),5 (box coords + class))
    cc = 0
    for target in targets:
        new_targets[cc,:target.shape[0]] = target #Each sample now always contains 2 targets (initialized to zero). This is done to make iteration of GTs easier for each sample
        cc += 1
    images_ = get_image_list_resized(images) #Tensor size: [16, 3, 608, 1088], Originally used in maskrcnn to padding varying sized images to same size. Not Sure why used here
    #images_ = images
    cts = torch.LongTensor(counts)
    # print(images_.shape)
    return images_, new_targets, cts, image_ids, whs

# if __name__== '__main__':
# #    from torchvision import transforms
#     dataset= DetectionDataset(root= folder, image_sets= 'train')
#     dataset_val= DetectionDataset(root= folder, image_sets= 'val')
#     print('train',len(dataset))
#     print('val',len(dataset_val))
