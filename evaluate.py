
""" 
    
    Adapted from:
    Modification by: Gurkirt Singh
    Modification started: 2nd April 2019
    large parts of this files are from many github repos
    mainly adopted from
    https://github.com/gurkirt/realtime-action-detection

    Please don't remove above credits and give star to these repos
    Licensed under The MIT License [see LICENSE for details]    

"""

import os
import pdb
import time, json
import socket
import getpass 
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.optim.lr_scheduler import MultiStepLR
from modules import utils
from modules.utils import str2bool
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from data import DetectionDataset, custum_collate
from models.retinanet_shared_heads import build_retinanet_shared_heads
from torchvision import transforms
from data.transforms import Resize
from train import validate

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
# anchor_type to be used in the experiment
parser.add_argument('--anchor_type', default='kmeans', help='kmeans or default')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet50', help='pretrained base model')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--multi_scale', default=False, type=str2bool,help='perfrom multiscale training')
parser.add_argument('--shared_heads', default=0, type=int,help='4 head layers')
parser.add_argument('--num_head_layers', default=4, type=int,help='0 mean no shareding more than 0 means shareing')
parser.add_argument('--use_bias', default=True, type=str2bool,help='0 mean no bias in head layears')
#  Name of the dataset only esad is supported
parser.add_argument('--dataset', default='esad', help='pretrained base model')
# Input size of image only 600 is supprted at the moment 
parser.add_argument('--min_size', default=600, type=int, help='Input Size for FPN')
parser.add_argument('--max_size', default=1000, type=int, help='Input Size for FPN')
#  data loading argumnets
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
# Number of worker to load data in parllel
parser.add_argument('--num_workers', '-j', default=8, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--optim', default='SGD', type=str, help='Optimiser type')
parser.add_argument('--loss_type', default='mbox', type=str, help='loss_type')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--eval_iters', default='90000', type=str, help='Chnage the lr @')

# Freeze batch normlisatio layer or not 
parser.add_argument('--fbn', default=True, type=bool, help='if less than 1 mean freeze or else any positive values keep updating bn layers')
parser.add_argument('--freezeupto', default=1, type=int, help='if 0 freeze or else keep updating bn layers')

# Evaluation hyperparameters
parser.add_argument('--iou_thresh', default=0.5, type=float, help='Evaluation threshold')
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.5, type=float, help='NMS threshold')
parser.add_argument('--topk', default=25, type=int, help='topk for evaluation')

# Progress logging
parser.add_argument('--log_iters', default=True, type=str2bool, help='Print the loss at each iteration')
parser.add_argument('--log_step', default=10, type=int, help='Log after k steps for text/tensorboard')
parser.add_argument('--tensorboard', default=False, type=str2bool, help='Use tensorboard for loss/evalaution visualization')

# Program arguments
parser.add_argument('--man_seed', default=1, type=int, help='manualseed for reproduction')
parser.add_argument('--multi_gpu', default=1, type=int, help='If  more than then use all visible GPUs by default only one GPU used ') 

# source or dstination directories
parser.add_argument('--data_root', default='/mnt/mercury-fast/datasets/', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
parser.add_argument('--save_root', default='/mnt/mercury-fast/datasets/', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
parser.add_argument('--model_dir', default='', help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/


## Parse arguments
args = parser.parse_args()

args = utils.set_args(args, 'test') # set directories and subsets fo datasets

## set random seeds and global settings
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
torch.cuda.manual_seed_all(args.man_seed)
torch.set_default_tensor_type('torch.FloatTensor')


def main():
    
    args.exp_name = utils.create_exp_name(args)

    args.save_root += args.dataset+'/'
    args.save_root = args.save_root+'cache/'+args.exp_name+'/'


    val_transform = transforms.Compose([ 
                        Resize(args.min_size, args.max_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means,std=args.stds)])

    val_dataset = DetectionDataset(args, train=False, image_sets=args.val_sets, 
                            transform=val_transform, full_test=False)

    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)

    args.data_dir = val_dataset.root
    args.num_classes = len(val_dataset.classes) + 1
    args.classes = val_dataset.classes
    args.head_size = 256
    
    net = build_retinanet_shared_heads(args).cuda()

    if args.multi_gpu>0:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
    net.eval()

    for iteration in args.eval_iters:
        args.det_itr = iteration
        log_file = open("{pt:s}/testing-{it:06d}-{date:%m-%d-%Hx}.log".format(pt=args.save_root, it=iteration, date=datetime.datetime.now()), "w", 10)
        log_file.write(args.exp_name + '\n')
        
        args.model_path = args.save_root + 'model_{:06d}.pth'.format(iteration)
        log_file.write(args.model_path+'\n')
    
        net.load_state_dict(torch.load(args.model_path))

        print('Finished loading model %d !' % iteration)
        # Load dataset
        val_data_loader = data_utils.DataLoader(val_dataset, int(args.batch_size), num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)

        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        net.eval() # switch net to evaluation mode
        
        mAP, ap_all, ap_strs , det_boxes = validate(args, net, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh)

        for ap_str in ap_strs:
            print(ap_str)
            log_file.write(ap_str+'\n')
        ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
        print(ptr_str)
        log_file.write(ptr_str)

        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()

    
if __name__ == '__main__':
    main()
