""" 

    Author: Vivek Singh and Gurkirt Singh 
    DATE: April 2020

    Parts of this files are from many github repos

    @gurkirt mostly from https://github.com/gurkirt/RetinaNet.pytorch.1.x

    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Which was adopated by: Ellis Brown, Max deGroot
    https://github.com/amdegroot/ssd.pytorch

    mainly adopted from
    https://github.com/gurkirt/realtime-action-detection

    maybe more but that is where I got these from
    
    Please don't remove above credits and give star to these repos
    Licensed under The MIT License [see LICENSE for details]

"""

import os
import time
import socket
import getpass 
import argparse
import datetime
import pdb
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from modules.solver import get_optim
from modules import utils
from modules.detection_loss import MultiBoxLoss, YOLOLoss, FocalLoss
from modules.evaluation import evaluate_detections
from modules.box_utils import decode, nms
from modules import  AverageMeter
from data import DetectionDataset, custum_collate
from torchvision import transforms
from data.transforms import Resize
from models.retinanet_shared_heads import build_retinanet_shared_heads

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def make_01(v):
       return 1 if v>0 else 0

parser = argparse.ArgumentParser(description='Training single stage FPN with OHEM, resnet as backbone')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet18', help='pretrained base model')
# Use LSTM
parser.add_argument('--append_lstm', default=True, type=str2bool, help='Append lstm layer before flattened retinanet predictor heads')
#parser.add_argument('--lstm_depth', default=198, type=int, help='Append lstm layer after FCN layers of retinaNet')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--multi_scale', default=False, type=str2bool,help='perfrom multiscale training')
parser.add_argument('--shared_heads', default=0, type=int,help='0 mean no sharing more than 0 means sharing')
parser.add_argument('--num_head_layers', default=4, type=int,help='4 head layers')
parser.add_argument('--use_bias', default=True, type=str2bool,help='0 mean no bias in head layears')
#  Name of the dataset only voc or coco are supported
parser.add_argument('--dataset', default='esad', help='pretrained base model')
# Input size of image only 600 is supprted at the moment 
parser.add_argument('--min_size', default=200, type=int, help='Input Size for FPN') #o: 600
parser.add_argument('--max_size', default=1080, type=int, help='Input Size for FPN')
#  data loading argumnets
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training') # o:16
# Number of worker to load data in parllel
parser.add_argument('--num_workers', '-j', default=4, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--optim', default='SGD', type=str, help='Optimiser type')
parser.add_argument('--resume', default=0, type=int, help='Resume from given iterations')
parser.add_argument('--max_iter', default=2, type=int, help='Number of training iterations') #o:9000
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--loss_type', default='focal', type=str, help='loss_type') #o:mbox
parser.add_argument('--milestones', default='6000,8000', type=str, help='Chnage the lr @')
parser.add_argument('--gammas', default='0.1,0.1', type=str, help='Gamma update for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')

# Freeze layers or not 
parser.add_argument('--fbn','--freeze_bn', default=True, type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
parser.add_argument('--freezeupto', default=1, type=int, help='layer group number in ResNet up to which needs to be frozen')

# Loss function matching threshold
parser.add_argument('--positive_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--negative_threshold', default=0.4, type=float, help='Min Jaccard index for matching')

# Evaluation hyperparameters
parser.add_argument('--intial_val', default=5000, type=int, help='Initial number of training iterations before evaluation')
parser.add_argument('--val_step', default=1000, type=int, help='Number of training iterations before evaluation')
parser.add_argument('--iou_thresh', default=0.30, type=float, help='Evaluation threshold') #For evaluation of val set, just check on AP50
parser.add_argument('--conf_thresh', default=0.05, type=float, help='Confidence threshold for evaluation')
parser.add_argument('--nms_thresh', default=0.45, type=float, help='NMS threshold')
parser.add_argument('--topk', default=50, type=int, help='topk for evaluation')

# Progress logging
parser.add_argument('--log_start', default=10, type=int, help='start loging after k steps for text/tensorboard') # Let initial ripples settle down
parser.add_argument('--log_step', default=1, type=int, help='Log every k steps for text/tensorboard')
parser.add_argument('--tensorboard', default=True, type=str2bool, help='Use tensorboard for loss/evalaution visualization')

# Program arguments
parser.add_argument('--man_seed', default=123, type=int, help='manualseed for reproduction')

parser.add_argument('--multi_gpu', default=True, type=str2bool, help='If  more than 0 then use all visible GPUs by default only one GPU used ') 

# Use CUDA_VISIBLE_DEVICES=0,1,4,6 to select GPUs to use
parser.add_argument('--data_root', default='../', help='Location to root directory fo dataset') # /mnt/mars-fast/datasets/
parser.add_argument('--save_root', default="..\\checkpoint\\", help='Location to save checkpoint models') # /mnt/sun-gamma/datasets/
# parser.add_argument('--model_dir', default='/mnt/sun-beta/vivek/weights/', help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/
parser.add_argument('--model_dir', default='../pretrain/resnet/', help='Location to where imagenet pretrained models exists') # /mnt/mars-fast/datasets/

# args.model_dir = ''
## Parse arguments
args = parser.parse_args()

args = utils.set_args(args) # set directories and subsets fo datasets

if args.tensorboard:
    from tensorboardX import SummaryWriter

## set random seeds and global settings
np.random.seed(args.man_seed)
torch.manual_seed(args.man_seed)
torch.cuda.manual_seed_all(args.man_seed)
torch.set_default_tensor_type('torch.FloatTensor')


def main():
    
    args.exp_name = utils.create_exp_name(args)
    args.save_root += args.dataset+'\\'
    args.data_root += args.dataset+'\\'
    args.save_root = args.save_root+'cache\\'+args.exp_name+'\\'

    if not os.path.isdir(args.save_root): #if save directory doesn't exist create it
        os.makedirs(args.save_root)

    source_dir = args.save_root+'source\\' # where to save the source
    utils.copy_source(source_dir) # make a copy of source files used for training as a snapshot

    print('\nLoading Datasets')

    train_transform = transforms.Compose([
                        Resize(args.min_size, args.max_size),  #w=min size (200) h=200*1080/1920 =112  if height > width, then image will be rescaled to (size * height / width, size)
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means, std=args.stds)])

    train_dataset = DetectionDataset(root= args.data_root, train=True, input_sets=['train/set1','train/set2'], transform=train_transform)
    print('Done Loading Dataset Train Dataset :::>>>\n',train_dataset.print_str)
    val_transform = transforms.Compose([ 
                        Resize(args.min_size, args.max_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means,std=args.stds)])
                        
    val_dataset = DetectionDataset(root= args.data_root, train=False, input_sets=['val/obj'], transform=val_transform, full_test=False)
    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)
    
    args.num_classes = len(train_dataset.classes) + 1
    args.classes = train_dataset.classes
    args.use_bias = args.use_bias>0
    args.head_size = 256
    
    net = build_retinanet_shared_heads(args).cuda() #The author has hardcoded the use of cuda so i can't use it on cpu.
    
    # print(net)
    args.multi_gpu = False
    if args.multi_gpu:
        print('\nLets do dataparallel\n')
        net = torch.nn.DataParallel(net)
 
    if args.fbn:
        if args.multi_gpu:
            net.module.backbone_net.apply(utils.set_bn_eval)
        else:
            net.backbone_net.apply(utils.set_bn_eval)
    
    optimizer, scheduler, solver_print_str = get_optim(args, net)

    train(args, net, optimizer, scheduler, train_dataset, val_dataset, solver_print_str)


def train(args, net, optimizer, scheduler, train_dataset, val_dataset, solver_print_str):
    
    args.start_iteration = 0
    if args.resume>100:
        args.start_iteration = args.resume
        args.iteration = args.start_iteration
        for _ in range(args.iteration-1):
            scheduler.step()
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.save_root, args.start_iteration)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.save_root, args.start_iteration)
        net.load_state_dict(torch.load(model_file_name))
        optimizer.load_state_dict(torch.load(optimizer_file_name))
        
    # anchors = anchors.cuda(0, non_blocking=True)
    if args.tensorboard:
        log_dir = args.save_root+'tensorboard-{date:%m-%d-%Hx}.log'.format(date=datetime.datetime.now())
        sw = SummaryWriter(log_dir)
    log_file = open(args.save_root+'training.text{date:%m-%d-%Hx}.txt'.format(date=datetime.datetime.now()), 'w', 1)
    log_file.write(args.exp_name+'\n')

    for arg in sorted(vars(args)):
        print(arg, getattr(args, arg))
        log_file.write(str(arg)+': '+str(getattr(args, arg))+'\n')
    log_file.write(str(net))
    log_file.write(solver_print_str)
    net.train()
    

    # loss counters
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    cls_losses = AverageMeter()

    # train_dataset = DetectionDatasetDatasetDatasetDatasetDataset(args, 'train', BaseTransform(args.input_dim, args.means, args.stds))

    log_file.write(train_dataset.print_str)
    log_file.write(val_dataset.print_str)
    print('Train-DATA :::>>>', train_dataset.print_str)
    print('VAL-DATA :::>>>', val_dataset.print_str)
    epoch_size = len(train_dataset) // args.batch_size
#    print('Training FPN on ', train_dataset.dataset,'\n')


    train_data_loader = data_utils.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers,
                                  shuffle=True, pin_memory=True, collate_fn=custum_collate, drop_last=True)
    
    
    val_data_loader = data_utils.DataLoader(val_dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, collate_fn=custum_collate)
  
    torch.cuda.synchronize()
    start = time.perf_counter()
    iteration = args.start_iteration
    eopch = 0
    num_bpe = len(train_data_loader)
    while iteration <= args.max_iter:
        for i, (images, gts, counts, _, _) in enumerate(train_data_loader):
            if iteration > args.max_iter:
                break
            iteration += 1
            
#            pdb.set_trace()
            epoch = int(iteration/num_bpe)
            images = images.cuda(0, non_blocking=True)
            gts = gts.cuda(0, non_blocking=True)
            counts = counts.cuda(0, non_blocking=True)
            # forward
            torch.cuda.synchronize()
            data_time.update(time.perf_counter() - start)

            # print(images.size(), anchors.size())
            optimizer.zero_grad()
            # pdb.set_trace()
            # print(gts.shape, counts.shape, images.shape)
            loss_l, loss_c = net(images, gts, counts)
            loss_l, loss_c = loss_l.mean() , loss_c.mean()
            loss = loss_l + loss_c

            loss.backward()
            optimizer.step()
            scheduler.step()

#            pdb.set_trace()
            loc_loss = loss_l.item()
            conf_loss = loss_c.item()
            
            if loc_loss>300:
                lline = '\n\n\n We got faulty LOCATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                log_file.write(lline)
                print(lline)
                loc_loss = 20.0
            if conf_loss>300:
                lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} \n\n\n'.format(loc_loss, conf_loss)
                log_file.write(lline)
                print(lline)
                conf_loss = 20.0
            
            # print('Loss data type ',type(loc_loss))
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)
            losses.update((loc_loss + conf_loss)/2.0)

            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()

            if iteration % args.log_step == 0 and iteration > args.log_start:
                if args.tensorboard:
                    sw.add_scalars('Classification', {'val': cls_losses.val, 'avg':cls_losses.avg},iteration)
                    sw.add_scalars('Localisation', {'val': loc_losses.val, 'avg':loc_losses.avg},iteration)
                    sw.add_scalars('Overall', {'val': losses.val, 'avg':losses.avg},iteration)
                    
                print_line = 'Itration [{:d}]{:06d}/{:06d} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) ' \
                             'average-loss {:.2f}({:.2f}) DataTime{:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format( epoch,
                              iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                              cls_losses.avg, losses.val, losses.avg, 10*data_time.val, 10*data_time.avg, 10*batch_time.val, 10*batch_time.avg)

                log_file.write(print_line+'\n')
                print(print_line)
                if iteration % (args.log_step*10) == 0:
                    print_line = args.exp_name
                    log_file.write(print_line+'\n')
                    print(print_line)


            if (iteration % args.val_step == 0 or iteration== args.intial_val or iteration == args.max_iter) and iteration>0:
                torch.cuda.synchronize()
                tvs = time.perf_counter()
                print('Saving state, iter:', iteration)
                torch.save(net.state_dict(), '{:s}/model_{:06d}.pth'.format(args.save_root, iteration))
                torch.save(optimizer.state_dict(), '{:s}/optimizer_{:06d}.pth'.format(args.save_root, iteration))
                net.eval() # switch net to evaluation mode
                mAP, ap_all, ap_strs, _ = validate(args, net, val_data_loader, val_dataset, iteration, iou_thresh=args.iou_thresh) #Finds only one IoU
                net.train()
                if args.fbn:
                    if args.multi_gpu:
                        net.module.backbone_net.apply(utils.set_bn_eval)
                    else:
                        net.backbone_net.apply(utils.set_bn_eval)

                for ap_str in ap_strs:
                    print(ap_str)
                    log_file.write(ap_str+'\n')
                ptr_str = '\nMEANAP:::=>'+str(mAP)+'\n'
                print(ptr_str)
                log_file.write(ptr_str)

                if args.tensorboard:
                    sw.add_scalar('mAP@0.5', mAP, iteration)
                    class_AP_group = dict()
                    for c, ap in enumerate(ap_all):
                        class_AP_group[args.classes[c]] = ap
                    sw.add_scalars('ClassAPs', class_AP_group, iteration)

                torch.cuda.synchronize()
                t0 = time.perf_counter()
                prt_str = '\nValidation TIME::: {:0.3f}\n\n'.format(t0-tvs)
                print(prt_str)
                log_file.write(prt_str)

    log_file.close()


def validate(args, net,  val_data_loader, val_dataset, iteration_num, iou_thresh=0.5):
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
            decoded_boxes, conf_data = net(images)

            conf_scores_all = activation(conf_data).clone()

            if print_time and val_itr%val_step == 0:
                torch.cuda.synchronize()
                tf = time.perf_counter()
                print('Forward Time {:0.3f}'.format(tf-t1))
            
            for b in range(batch_size):
                width, height = wh[b][0], wh[b][1]
                gt = targets[b, :batch_counts[b]].numpy()
                gt_boxes.append(gt)
                # decoded_boxes = decode(loc_data[b], anchors).clone()
                conf_scores = conf_scores_all[b]
                #Apply nms per class and obtain the results
                decoded_boxes_b = decoded_boxes[b]
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

                    for ik in range(boxes.shape[0]):
                        boxes[ik, 0] = max(0, boxes[ik, 0])
                        boxes[ik, 2] = min(width, boxes[ik, 2])
                        boxes[ik, 1] = max(0, boxes[ik, 1])
                        boxes[ik, 3] = min(height, boxes[ik, 3])
                    
                    cls_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=True)
                    det_boxes[cl_ind-1].append(cls_dets)
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

    print('Evaluating detections for itration number ', iteration_num)
    return evaluate_detections(gt_boxes, det_boxes, val_dataset.classes, iou_thresh=iou_thresh)

if __name__ == '__main__':
    main()
