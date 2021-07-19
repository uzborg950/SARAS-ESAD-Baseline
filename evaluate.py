
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
from modules import frame_utils
from modules import  AverageMeter
from data import DetectionDataset, custom_collate
from models.retinanet_shared_heads import build_retinanet_shared_heads
from torchvision import transforms
from data.transforms import Resize
from functools import partial

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
#Predicted Bounding Box visualisation
parser.add_argument('--generate_frames', default=True,type=str2bool, help='Generate frame containing GT and predicted bounding boxes')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet18', help='pretrained base model')
# Multi-Task Surgical Phase Detection
parser.add_argument('--predict_surgical_phase', default=False, type=str2bool, help='predict surgical phase as well')
parser.add_argument('--num_phases', default=4, type=int, help='Total number of phases')
# Use Time Distribution for CNN backbone
parser.add_argument('--time_distributed_backbone', default=False, type=str2bool, help='Make backbone time distributed (Apply the same backbone weights to a number of timesteps')
parser.add_argument('--temporal_slice_timesteps', default=4, type=int, help='Number of timesteps/frame comprising a temporal slice')
# Use LSTM
parser.add_argument('--append_temporal_net', default=False, type=str2bool, help='Append temporal model after FPN, before predictor conv head')
parser.add_argument('--convlstm_layers', default=1, type=int, help='Number of stacked convlstm layers')
parser.add_argument('--temporal_net_layers', default=2, type=int, help='Number of temporal net layers (each layer = ConvLSTM(s) + Conv2d + batch norm + relu)')
parser.add_argument('--num_truncated_iterations', default=1, type=int, help='Truncate iterations during BPTT to down-scale computation graph')
parser.add_argument('--grad_accumulate_iterations', default=1, type=int, help='Accumulate gradients accross mini-batches upto the given number of iterations')
#parser.add_argument('--lstm_depth', default=128, type=int, help='Append lstm layer after FCN layers of retinaNet')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--multi_scale', default=False, type=str2bool,help='perfrom multiscale training')
parser.add_argument('--shared_heads', default=0, type=int,help='4 head layers')
parser.add_argument('--num_head_layers', default=4, type=int,help='0 mean no shareding more than 0 means shareing')
parser.add_argument('--use_bias', default=True, type=str2bool,help='0 mean no bias in head layears')
#  Name of the dataset only esad is supported
parser.add_argument('--dataset', default='esad', help='pretrained base model')
# Input size of image only 600 is supprted at the moment
parser.add_argument('--original_width', default=1920, type=int, help='Actual width of input')
parser.add_argument('--original_height', default=1080, type=int, help='Actual height of input')
parser.add_argument('--min_size', default=200, type=int, help='Input Size for FPN')
parser.add_argument('--max_size', default=1080, type=int, help='Input Size for FPN')
#  data loading argumnets
parser.add_argument('--shifted_mean', default=False, type=str2bool, help='Shift mean and std dev during normalisation')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
# Number of worker to load data in parllel
parser.add_argument('--num_workers', '-j', default=8, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--optim', default='SGD', type=str, help='Optimiser type')
parser.add_argument('--loss_type', default='focal', type=str, help='loss_type')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--eval_iters', default='9000', type=str, help='Chnage the lr @')#5000,6000,7000,9000

# Freeze batch normlisatio layer or not 
parser.add_argument('--fbn', default=True, type=bool, help='if less than 1 mean freeze or else any positive values keep updating bn layers')
parser.add_argument('--freezeupto', default=1, type=int, help='if 0 freeze or else keep updating bn layers')

# Evaluation hyperparameters
parser.add_argument('--iou_threshs', default='0.1,0.3,0.5', type=str, help='Evaluation thresholds')
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
parser.add_argument('--data_root', default='../', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
parser.add_argument('--save_root', default='..\\checkpoint\\', help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
parser.add_argument('--model_dir', default='../pretrain/resnet/', help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/


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

    args.data_root += args.dataset+'\\'
    args.save_root += args.dataset+'\\'
    args.save_root += 'cache\\'+args.exp_name+'\\'


    val_transform = transforms.Compose([ 
                        Resize(args.min_size, args.max_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means,std=args.stds)])
    if True: # while validating
        val_dataset = DetectionDataset(root= args.data_root, train=False, input_sets=['val/obj'], transform=val_transform, full_test=True, include_phase=args.predict_surgical_phase)
    else: # while testing
        val_dataset = DetectionDataset(root= args.data_root, train=False, input_sets=['testC'], transform=val_transform, full_test=True, include_phase=args.predict_surgical_phase)

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
    
    if len(args.iou_threshs)>2:
        args.iou_threshs = [float(th) for th in args.iou_threshs.split(',')]
    else:
        args.iou_threshs = [th for th in np.arange(0.05,0.951,0.05)]

    overal_json_object = {}
    for iteration in args.eval_iters:
        args.det_itr = iteration
        args.model_path = args.save_root + 'model_{:06d}.pth'.format(iteration)
        
        if not os.path.isfile(args.model_path):
            continue

        log_file = open("{pt:s}/testing-{it:06d}.log".format(pt=args.save_root, it=iteration), "w", 1)
        log_file.write(args.exp_name + '\n')
        submission_file = open("{pt:s}/submission.txt".format(pt=args.save_root), "w", 1)
        log_file.write(args.model_path+'\n')

        net.load_state_dict(torch.load(args.model_path))

        print('Finished loading model %d !' % iteration)
        # Load dataset
        val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size if not args.time_distributed_backbone else args.batch_size + args.temporal_slice_timesteps - 1, num_workers=args.num_workers,
                                                shuffle=False, pin_memory=True, collate_fn=partial(custom_collate, timesteps = args.batch_size if not args.time_distributed_backbone else args.batch_size + args.temporal_slice_timesteps - 1))

        # evaluation
        torch.cuda.synchronize()
        tt0 = time.perf_counter()
        log_file.write('Testing net \n')
        net.eval() # switch net to evaluation mode
        

        # print(args.iou_threshs)
        result_list = validate(args, net, val_data_loader, val_dataset, iteration, submission_file, args.iou_threshs) # finds APs for a number of IoUs
       
        for result in result_list:
            [iou_thresh, mAP, ap_all, ap_strs] = result 
            
            ptr_str = '\n\nIOU Threshold: {:0.2f}:: \n'.format(iou_thresh)
            print(ptr_str)
            log_file.write(ptr_str)

            for ap_str in ap_strs:
                print(ap_str)
                log_file.write(ap_str+'\n')
            
            ptr_str = '\nMEANAP:::=>{:0.4f}\n'.format(mAP*100)
            print(ptr_str)
            log_file.write(ptr_str)
        
        ptr_str = 'printing the on mAP results again \n'
        print(ptr_str)
        log_file.write(ptr_str)
        
        thmap_dict = {'30':0,'10':0,'50':0,'mean':0}
        summap = 0
        for result in result_list:
            [iou_thresh, mAP, ap_all, ap_strs] = result 
            thstr = str(int(iou_thresh*100))
            if thstr in thmap_dict:
                thmap_dict[thstr] = mAP*100
                summap += mAP*100
            ptr_str = '\nIOUTH : mAP :: {:0.2f} : {:0.2f}\n'.format(iou_thresh,mAP*100)
            print(ptr_str)
            log_file.write(ptr_str)
            
        thmap_dict['mean'] = summap/3.0
        overal_json_object[str(int(iteration))] = thmap_dict
        torch.cuda.synchronize()
        print('Complete set time {:0.2f}'.format(time.perf_counter() - tt0))
        log_file.close()

    result_name = '../results/test-frameAP-{:s}-{:s}-{:d}'.format(args.loss_type, args.basenet, args.min_size)

    with open(result_name+'.json', 'w+') as f:
        json.dump(overal_json_object,f)
    
    fid = open(result_name+'.txt', 'w+')
    fid.write('{:s} {:s} {:d}\n'.format(args.loss_type, args.basenet, args.min_size))
    for iter in sorted(overal_json_object.keys()):
        res = overal_json_object[iter]
        fid.write('{:s} {:0.1f} {:0.1f} {:0.1f} {:0.1f}\n'.format(iter, res['10'], res['30'], res['50'], res['mean']))
    fid.close()



def validate(args, net,  val_data_loader, val_dataset, iteration_num, submission_file, iou_threshs=[0.2,0.25,0.3,0.4,0.5,0.75]):
    """Test a FPN network on an image database."""
    print('Validating at ', iteration_num)
    num_images = len(val_dataset)
    num_classes = args.num_classes
    
    det_boxes = [[] for _ in range(num_classes-1)]
    gt_boxes = []
    print_time = True
    val_step = 20
    count = 0
    torch.cuda.synchronize()
    ts = time.perf_counter()
    activation = nn.Sigmoid().cuda()
    if args.loss_type == 'mbox':
        activation = nn.Softmax(dim=2).cuda()

    dict_for_json_dump = {}

    with torch.no_grad():
        for val_itr, (images, targets, batch_counts, img_indexs, wh) in enumerate(val_data_loader):

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)

            images = images.cuda(0, non_blocking=True)

            if args.time_distributed_backbone:
                _, channels, height, width = images.shape
                images = construct_temporal_batches(images, args.batch_size, args.temporal_slice_timesteps)
                gts, counts = generate_temporal_gts(gts, batch_counts, args.batch_size, args.temporal_slice_timesteps)


            decoded_boxes, conf_data = net(images)

            conf_scores_all = activation(conf_data).clone()

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))

            if args.time_distributed_backbone: #merge batch size and timesteps for analysis operations (generate frames) of batch
                images = images.view(images.shape[0] * images.shape[1], images.shape[2], images.shape[3],
                                     images.shape[4])

            for b in range(batch_size):
                image_path = val_dataset.ids[img_indexs[b]][0]
                image_name = image_path.split('/')[-1]
                # print(image_name)
                width, height = wh[b][0], wh[b][1]
                gt = targets[b, :batch_counts[b]].numpy()
                gt_boxes.append(gt)
                # decoded_boxes = decode(loc_data[b], anchors).clone()
                conf_scores = conf_scores_all[b]
                #Apply nms per class and obtain the results
                decoded_boxes_b = decoded_boxes[b]


                if args.generate_frames:
                    ax = frame_utils.generate_frame(images[b].permute(1, 2, 0), targets[b, :, :],
                                                   None, None,gt=True,show=False)
                for cl_ind in range(1, num_classes):
                    # pdb.set_trace()
                    scores = conf_scores[:, cl_ind].squeeze()
                    if args.loss_type == 'yolo':
                        scores = conf_scores[:, cl_ind].squeeze() * conf_scores[:, 0].squeeze() * 5.0
                    c_mask = scores.gt(args.conf_thresh)  # greater than minmum threshold
                    scores = scores[c_mask].squeeze()
                    # print('scores size',c_mask.sum())
                    if scores.dim() == 0:
                        # print(len(''), ' dim ==0 ')
                        det_boxes[cl_ind - 1].append(np.asarray([]))
                        continue
                    # boxes = decoded_boxes_b.clone()
                    l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes_b)
                    boxes = decoded_boxes_b[l_mask].clone().view(-1, 4)
                    # idx of highest scoring and non-overlapping boxes per class
                    ids, counts = nms(boxes, scores, args.nms_thresh, args.topk*20)  # idsn - ids after nms
                    scores = scores[ids[:min(args.topk,counts)]].cpu().numpy()
                    # pick = min(scores.shape[0], 20)
                    # scores = scores[:pick]
                    boxes = boxes[ids[:min(args.topk,counts)]].cpu().numpy()
                    if args.generate_frames and boxes.shape[0] != 0:
                        ax = frame_utils.generate_frame(images[b].permute(1, 2, 0), torch.cat((torch.tensor(boxes), torch.tensor(cl_ind-1).repeat( boxes.shape[0],1)), axis=1),
                                                    None, ax, show=False)
                    for ik in range(boxes.shape[0]):

                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])
                        write_string = '{:s} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:0.6f} {:d}\n'.format(image_name, 
                                        boxes[ik, 0]/width, boxes[ik, 1]/height, boxes[ik, 2]/width, boxes[ik, 3]/height, scores[ik], cl_ind-1)
                        submission_file.write(write_string)

                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind-1].append(cls_dets)
                if args.generate_frames:
                    frame_utils.save_image(images[b].permute(1, 2, 0), ax, filename=str((val_itr * batch_size) + b+1) + ".jpg")
                count += 1
            
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('im_detect: {:d}/{:d} time taken {:0.3f}'.format(count, num_images, te-ts))
                torch.cuda.synchronize()
                ts = time.perf_counter()
            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                te = time.perf_counter()
                print('NMS stuff Time {:0.3f}'.format(te - tf))


    submission_file.close()
    print('Evaluating detections for itration number ', iteration_num)
    return_list = []
    for iou_thresh in iou_threshs:
        mAP, ap_all, ap_strs , _ = evaluate_detections(gt_boxes, det_boxes, val_dataset.classes, iou_thresh=iou_thresh)
        return_list.append([iou_thresh, mAP, ap_all, ap_strs])
    return return_list 

def construct_temporal_batches(images, batch_size, timesteps):
    _, channels, height, width = images.shape
    images_td = torch.tensor([]).cuda()
    for seq_idx in range(batch_size):
        seq = images[seq_idx:seq_idx + timesteps, :,:,:]
        images_td = torch.cat((images_td, torch.unsqueeze(seq, 0)), 0)
    return images_td

def generate_temporal_gts(gts, counts, batch_size, timesteps):
    timestep_gts = torch.tensor([], dtype=torch.int64).cuda()
    timestep_counts = torch.tensor([], dtype=torch.int64).cuda()
    for seq_idx in range(batch_size):
        timestep_gts = torch.cat((timestep_gts, gts[seq_idx:seq_idx + timesteps]))
        timestep_counts = torch.cat((timestep_counts, counts[seq_idx:seq_idx + timesteps]))
    return timestep_gts, timestep_counts
if __name__ == '__main__':
    main()
