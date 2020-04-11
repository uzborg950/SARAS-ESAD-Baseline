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


#folder= '/mnt/sun-beta/saras_data/MIDL_dataset/'



def read_file(path):
    with open(path, 'r') as f:
        data1= f.readlines()
    
    if data1: 
        data2= [sam.split(' ') for sam in data1]
        data3=[]
        for i in data2:
            j= [float(sam) for sam in i]
            k= [j[1], j[2], j[3], j[4], j[0]]
            data3.append(k)
    else:
        data3= None
    return(data3)


def read_labels(data_files):
    labels=[]
    for txt_file in data_files:
        label= read_file(txt_file)
        if label is not None:
            img_path= txt_file.replace('.txt', '.jpg')
            labels.append([img_path, label])
    return labels

#%%

def read_train(path):
    folders= ['set1', 'set2']
    all_files=[]
    for path1 in folders:
        path2= path+ '/train/' + path1
        data_files= glob.glob(path2+'/*.txt')
        all_files.extend(data_files)
        
    labels= read_labels(all_files)
    return(labels)
    
            
def make_object_lists(rootpath, subset='train'):
    '''
    subset has two options: 
    train: to read the training images 
    val: to read the vaidation images
    '''

    with open(rootpath+'/train/obj.names', 'r') as fil:
        cls_list= fil.read().split('\n')
    
    if subset== 'train':
        final_labels= read_train(rootpath)
    elif subset== 'val':
        path= rootpath+'/val/obj/*.txt'
        file_paths= glob.glob(path)
        final_labels= read_labels(file_paths)
        
    return(cls_list, final_labels)


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class DetectionDataset(data.Dataset):
    """Detection Dataset class for pytorch dataloader"""

    def __init__(self, root, train=True, image_sets='train', transform=None, anno_transform=None, full_test=False):

#        self.dataset = args.dataset
        self.train = train
#        self.root = args.data_root
        self.root= root
        self.image_sets = image_sets
        self.transform = transform
        self.anno_transform = anno_transform
        self.ids = list()
        self.classes, self.ids = make_object_lists(self.root, subset=image_sets)
        self.print_str= ''
        self.max_targets = 20
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        annot_info = self.ids[index]
        
        img_path = annot_info[0]
        bbox_info = np.array(annot_info[1])   # boxes should be in x1 y1 x2 y2 format
        labels= bbox_info[:,4]
#        boxes= bbox_info[:,:4]
        x1= bbox_info[:,0]-bbox_info[:,2]/2
        y1= bbox_info[:,1]-bbox_info[:,3]/2
        x2= bbox_info[:,0]+ bbox_info[:,2]/2
        y2= bbox_info[:,1] + bbox_info[:,3]/2
        boxes= np.hstack([np.expand_dims(x1, axis=1), np.expand_dims(y1, axis=1),
                          np.expand_dims(x2, axis=1), np.expand_dims(y2, axis=1)])


        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

    
        # print(img.size)
#        if self.train and np.random.random() < 0.5:
#            img = img.transpose(Image.FLIP_LEFT_RIGHT)
#            w = boxes[:, 2] - boxes[:, 0]
#            boxes[:, 0] = 1 - boxes[:, 2] # boxes should be in x1 y1 x2 y2 [0,1] format 
#            boxes[:, 2] = boxes[:, 0] + w # boxes should be in x1 y1 x2 y2 [0,1] format
        

#         print(img.size, wh)
        if self.transform is not None:
            img = self.transform(img)
#        
        _, height, width = img.shape
#        # print(img.shape, wh)
        wh = [width, height, orig_w, orig_h]
#        # print(wh)
        boxes[:, 0] *= width # width x1
        boxes[:, 2] *= width # width x2
        boxes[:, 1] *= height # height y1
        boxes[:, 3] *= height # height y2
#
        targets = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, targets, index, wh
        
#        return(boxes, labels, img)

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

if __name__== '__main__':
#    from torchvision import transforms
    dataset= DetectionDataset(root= folder, image_sets= 'train')
    dataset_val= DetectionDataset(root= folder, image_sets= 'val')
    print('train',len(dataset))
    print('val',len(dataset_val))
    
#    boxes, label, img= dataset[2868]
#    draw= ImageDraw.Draw(img)
#    for box in boxes:
#        draw.rectangle(tuple(box.astype(np.int32)), outline= 'red', width=5)
#    img.show()
#    val_data= DetectionDataset(root= folder, image_sets= 'val')    
    
#    classes, data= make_object_lists(folder, subset='train')
#    create_data_csv_files(classes, data, filename='train_data.csv')
    




