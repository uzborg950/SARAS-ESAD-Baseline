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

import argparse
import datetime
import os
import time
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils
from torchvision import transforms
from functools import partial


from data import DetectionDataset, custom_collate
from data.transforms import Resize
from models.retinanet_shared_heads import build_retinanet_shared_heads
from modules import AverageMeter
from modules import utils
from modules.box_utils import nms
from modules.evaluation import evaluate_detections
from modules.solver import get_optim


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def make_01(v):
       return 1 if v>0 else 0

parser = argparse.ArgumentParser(description='Training single stage FPN with Focal, resnet as backbone')
# Name of backbone networ, e.g. resnet18, resnet34, resnet50, resnet101 resnet152 are supported 
parser.add_argument('--basenet', default='resnet18', help='pretrained base model')
#Binary classification loss
parser.add_argument('--bin_loss', default=True, type=str2bool, help='Include binary classification loss (object/background')
# Multi-Task Surgical Phase Detection
parser.add_argument('--predict_surgical_phase', default=False, type=str2bool, help='predict surgical phase as well')
parser.add_argument('--num_phases', default=4, type=int, help='Total number of phases')
# Use Time Distribution for CNN backbone
parser.add_argument('--time_distributed_backbone', default=True, type=str2bool, help='Make backbone time distributed (Apply the same backbone weights to a number of timesteps')
parser.add_argument('--temporal_slice_timesteps', default=1, type=int, help='Number of timesteps/frame comprising a temporal slice')
# Use ConvLSTM
parser.add_argument('--append_cls_temporal_net', default=True, type=str2bool, help='Append cls temporal model after FPN, before cls predictor conv head')
parser.add_argument('--append_reg_temporal_net', default=True, type=str2bool, help='Append regression temporal model after FPN, before regression predictor conv head')
parser.add_argument('--convlstm_layers', default=1, type=int, help='Number of stacked convlstm layers')
parser.add_argument('--temporal_net_layers', default=2, type=int, help='Number of temporal net layers (each layer = ConvLSTM(s) + Conv2d + batch norm + relu)')
parser.add_argument('--truncate_bptt', default=True, type=str2bool, help='Truncate iterations during BPTT to down-scale computation graph') #True
parser.add_argument('--k1', default=1, type=int, help='Number of forward pass timesteps between updates')
parser.add_argument('--k2', default=2, type=int, help='Number of timesteps to apply bptt to')
parser.add_argument('--grad_accumulate_iterations', default=1, type=int, help='Accumulate gradients accross mini-batches upto the given number of iterations') #100
parser.add_argument('--enable_variable_grad_accumulation', default=False, type=str2bool, help='Configure gradient accumulation across full train sets of variable sizes')
parser.add_argument('--override_optimization_steps_tbptt', default=False, type=str2bool, help='Override when optimization is run in TBPTT')#False
parser.add_argument('--overrided_optimization_steps', default=10, type=int, help='Override when optimization is run in TBPTT')#False
parser.add_argument('--reset_hidden_every_step', default=False, type=str2bool, help='Reset hidden state at every optimizer step')#False
#parser.add_argument('--lstm_depth', default=198, type=int, help='Append lstm layer after FCN layers of retinaNet')
# if output heads are have shared features or not: 0 is no-shareing else sharining enabled
parser.add_argument('--multi_scale', default=False, type=str2bool,help='perfrom multiscale training')
parser.add_argument('--shared_heads', default=0, type=int,help='0 mean no sharing more than 0 means sharing')
parser.add_argument('--num_head_layers', default=4, type=int,help='4 head layers')
parser.add_argument('--use_bias', default=True, type=str2bool,help='0 mean no bias in head layears')
#  Name of the dataset only voc or coco are supported
parser.add_argument('--dataset', default='esad', help='pretrained base model')
# Input size of image only 600 is supprted at the moment
parser.add_argument('--original_width', default=1920, type=int, help='Actual width of input')
parser.add_argument('--original_height', default=1080, type=int, help='Actual height of input')
parser.add_argument('--min_size', default=200, type=int, help='Input Size for FPN') #o: 600
parser.add_argument('--max_size', default=1080, type=int, help='Input Size for FPN')
#  data loading argumnets
parser.add_argument('--shifted_mean', default=False, type=str2bool, help='Shift mean and std dev during normalisation')
parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training') # o:16
parser.add_argument('--shuffle', default=False, type=str2bool, help='Shuffle training data') #False

# Number of worker to load data in parllel
parser.add_argument('--num_workers', '-j', default=4, type=int, help='Number of workers used in dataloading')
# optimiser hyperparameters
parser.add_argument('--optim', default='SGD', type=str, help='Optimiser type')#SGD
parser.add_argument('--load_non_strict_pretrained', default=False, type=str2bool, help='Load pretrained weights of partial model')
parser.add_argument('--freeze_cls_heads', default=False, type=str2bool, help='Freeze training of classification heads (excluding LSTM)')
parser.add_argument('--freeze_reg_heads', default=False, type=str2bool, help='Freeze training of box regression heads (excluding LSTM)')
parser.add_argument('--freeze_backbone', default=False, type=str2bool, help='Freeze training of resentFPN')
parser.add_argument('--pretrained_iter', default=50525, type=int, help='Iteration at which pretraining was stopped') #17000
parser.add_argument('--resume', default=0, type=int, help='Resume from given iterations')
parser.add_argument('--max_epochs', default=90, type=int, help='Number of epochs to run for')
parser.add_argument('--max_iter', default=422910, type=int, help='Number of training iterations') #b=16 105750; b=8 211500 ;b=4: 422910; b=2 845730
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, help='initial learning rate') #0.01
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--loss_type', default='focal', type=str, help='loss_type')  # o:mbox
parser.add_argument('--milestones', default='46990,93980,140970', type=str, help='Chnage the lr @')#b=16  '11750, 23500, 35250';b=8 '23500,47000,70500'; b=4 '46990,93980,140970'; b=2 '93970, 187940, 281910'
parser.add_argument('--gammas', default='0.1,0.1,0.1', type=str, help='Gamma update for SGD')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for SGD')

# Freeze layers or not 
parser.add_argument('--fbn','--freeze_bn', default=True, type=str2bool, help='freeze bn layers if true or else keep updating bn layers')
parser.add_argument('--freezeupto', default=1, type=int, help='layer group number in ResNet up to which needs to be frozen') # 1

# Loss function matching threshold
parser.add_argument('--positive_threshold', default=0.6, type=float, help='Min Jaccard index for matching')
parser.add_argument('--negative_threshold', default=0.4, type=float, help='Min Jaccard index for matching')

# Evaluation hyperparameters
parser.add_argument('--intial_val', default=4699, type=int, help='Initial number of training iterations before evaluation')
parser.add_argument('--val_step', default=4699, type=int, help='Number of training iterations before evaluation') #b=16, 1ep= 1175it , total= 18800 .   b=8, 1ep=2350 it, total= 18800. b=4, 1ep=4699 total=18796. b=2, 1ep=9397 total=18794
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

    train_dataset = DetectionDataset(root= args.data_root, train=True, input_sets=['train/set1','train/set2'], transform=train_transform, include_phase=args.predict_surgical_phase, batch=get_data_loader_batch_size(
            args))
    print('Done Loading Dataset Train Dataset :::>>>\n',train_dataset.print_str)
    val_transform = transforms.Compose([ 
                        Resize(args.min_size, args.max_size),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=args.means,std=args.stds)])
                        
    val_dataset = DetectionDataset(root= args.data_root, train=False, input_sets=['val/obj'], transform=val_transform, full_test=False, include_phase=args.predict_surgical_phase)
    print('Done Loading Dataset Validation Dataset :::>>>\n',val_dataset.print_str)

    args.train_set_sizes = train_dataset.input_set_sizes
    args.num_classes = len(train_dataset.classes) + 1
    args.classes = train_dataset.classes
    args.use_bias = args.use_bias>0
    args.head_size = 256
    
    net = build_retinanet_shared_heads(args).cuda() #The author has hardcoded the use of cuda so i can't use it on cpu.
    
    # print(net)
    #args.multi_gpu = False
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


def generate_temporal_gts(gts, counts, batch_size, timesteps):
    timestep_gts = torch.tensor([], dtype=torch.int64)
    timestep_counts = torch.tensor([], dtype=torch.int64)
    for seq_idx in range(batch_size):
        timestep_gts = torch.cat((timestep_gts, gts[seq_idx:seq_idx + timesteps]))
        timestep_counts = torch.cat((timestep_counts, counts[seq_idx:seq_idx + timesteps]))
    return timestep_gts, timestep_counts


def train(args, net, optimizer, scheduler, train_dataset, val_dataset, solver_print_str):
    
    args.start_iteration = 0
    if args.load_non_strict_pretrained:
        model_file_name = '{:s}/model_{:06d}.pth'.format(args.save_root, args.pretrained_iter)
        optimizer_file_name = '{:s}/optimizer_{:06d}.pth'.format(args.save_root, args.pretrained_iter)
        load_model_state_dict(model_file_name, net)
        #load_optim_state_dict(optimizer, optimizer_file_name)

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
    phase_losses = AverageMeter()

    # train_dataset = DetectionDatasetDatasetDatasetDatasetDataset(args, 'train', BaseTransform(args.input_dim, args.means, args.stds))

    log_file.write(train_dataset.print_str)
    log_file.write(val_dataset.print_str)
    print('Train-DATA :::>>>', train_dataset.print_str)
    print('VAL-DATA :::>>>', val_dataset.print_str)
    batch_size = args.batch_size
    epoch_size = len(train_dataset) // batch_size
#    print('Training FPN on ', train_dataset.dataset,'\n')


    train_data_loader = data_utils.DataLoader(train_dataset, get_data_loader_batch_size(args), num_workers=args.num_workers,
                                              shuffle=args.shuffle, pin_memory=True, collate_fn=partial(custom_collate, timesteps =get_data_loader_batch_size(
            args)), drop_last=True)

    
    val_data_loader = data_utils.DataLoader(val_dataset, get_data_loader_batch_size(args), num_workers=args.num_workers,
                                            shuffle=args.shuffle, pin_memory=True, collate_fn=partial(custom_collate, timesteps =get_data_loader_batch_size(
            args)))
  
    torch.cuda.synchronize()
    start = time.perf_counter()
    iteration = args.start_iteration
    epoch = 0
    num_bpe = len(train_data_loader)
    train_set_idx = 0

    loss = torch.zeros(1, requires_grad=True).cuda()
    images_td_batch = torch.tensor([])
    gts_td_batch =torch.tensor([])
    counts_td_batch = torch.tensor([])
    batch_fill_count = 0
    accumulation_counter = 0
    current_cls_loss = {}
    step_counter = 0
    grad_accumulate_iterations = args.grad_accumulate_iterations
    if args.truncate_bptt:
        grad_accumulate_iterations = args.k1
    k2 = args.k2
    retain_graph = args.k1 < args.k2 and args.truncate_bptt

    if args.enable_variable_grad_accumulation:
        args.train_set_sizes = [int(math.floor(size/get_data_loader_batch_size(args))) for size in args.train_set_sizes]
        grad_accumulate_iterations = args.train_set_sizes[train_set_idx]
    current_reg_hidden_states = None
    current_cls_hidden_states = None
    states = []
    torch.autograd.set_detect_anomaly(True)
    data_store = []
    while iteration <= args.max_iter or epoch < args.max_epochs:
        epoch +=1
        for i, (images, gts, counts, image_ids, _) in enumerate(train_data_loader):
            iteration += 1
            #if iteration % 1001 ==0: #Limit timesteps to 1000 for quickly testing learning ability of network
            #    break
            '''
            if( args.time_distributed_backbone and batch_fill_count < batch_size):
                #This hasn't been adapted for phase prediction included gts
                images_td_batch = cat_timestep(images, images_td_batch)
                gts_td_batch = cat_timestep(gts, gts_td_batch)
                counts_td_batch = cat_timestep(counts, counts_td_batch)
                batch_fill_count += 1
                if( batch_fill_count < batch_size):
                    continue
                images = images_td_batch
                gts = images_td_batch
                counts = counts_td_batch
            '''




            if args.time_distributed_backbone:
                _, channels, height, width = images.shape
                images = construct_temporal_batches(images, batch_size, args.temporal_slice_timesteps)
                #gts, counts = generate_temporal_gts(gts, counts, batch_size, args.temporal_slice_timesteps)

#            pdb.set_trace()
            #epoch = int(iteration/num_bpe)
            images = images.cuda(0, non_blocking=True)
            gts = gts.cuda(0, non_blocking=True)
            counts = counts.cuda(0, non_blocking=True)
            # forward
            torch.cuda.synchronize()
            accumulation_counter += 1
            data_time.update(time.perf_counter() - start)

            if args.truncate_bptt:
                if len(data_store) > k2:
                    del data_store[0]
                data_store.append((images.cpu(), gts.cpu(), counts.cpu()))

            # print(images.size(), anchors.size())

            # pdb.set_trace()
            # print(gts.shape, counts.shape, images.shape)

            if args.truncate_bptt and states:
                current_cls_hidden_states, current_reg_hidden_states = detach_ith_states(states, -1)

            #print("printing images size:")
            #print(images.shape[0] * images.shape[1])
            loss_l, loss_c, loss_p, new_reg_hidden_states, new_cls_hidden_states,full_init_reg_hidden_states, full_init_cls_hidden_states = net(images, gts, counts, reg_hidden_states=current_reg_hidden_states, cls_hidden_states=current_cls_hidden_states)

            loss_l, loss_c = loss_l.mean(), loss_c.mean()

            if args.predict_surgical_phase:
                loss_p = loss_p.mean()
                loss_t = loss_l + loss_c + loss_p  # + loss
            else:
                loss_t = loss_l + loss_c #+ loss

            loss = loss_t#loss + loss_t / grad_accumulate_iterations
            #if count_update - args.num_truncated_iterations == 500:
            #    count_update = 0
            #    loss = torch.zeros(1, requires_grad=True, device='cuda:0')

            #if(iteration % 200 == 0):
            #    scheduler.reduce_lr()

            torch.cuda.synchronize()
            if not states: #TODO TEST DATA FLOW OF init_reg_hidden_states, init_cls_hidden_states and append to states
                init_states, detached_states = get_init_and_detached_states(full_init_cls_hidden_states, full_init_reg_hidden_states)

                states.append((None, (init_states[0], init_states[1])))
                states.append(((detached_states[0], detached_states[1]),(new_reg_hidden_states, new_cls_hidden_states)))
            else:
                states.append(((current_reg_hidden_states, current_cls_hidden_states),(new_reg_hidden_states, new_cls_hidden_states)))

            while len(states) > k2:
                del states[0]


            if should_backprop(accumulation_counter, grad_accumulate_iterations, step_counter, train_set_idx, args): #Remember: can't use variable grad with trunc bptt

                optimizer.zero_grad()

                loss.backward(retain_graph=retain_graph)

                if args.truncate_bptt:
                    for i in range(k2-1):
                        if len(states) == 1 or states[-i-2][0] is None:
                            break
                        reg_hidden_states, cls_hidden_states = states[-i-1][0]
                        reg_hidden_states_prev, cls_hidden_states_prev = states[-i-2][1]
                        for reg_hidden, cls_hidden,reg_hidden_prev, cls_hidden_prev in zip(reg_hidden_states, cls_hidden_states, reg_hidden_states_prev, cls_hidden_states_prev):
                            for reg_temp_layer, cls_temp_layer,reg_temp_layer_prev, cls_temp_layer_prev in zip(reg_hidden, cls_hidden,reg_hidden_prev, cls_hidden_prev):
                                for reg_temp_block, cls_temp_block, reg_temp_block_prev, cls_temp_block_prev in zip(
                                        reg_temp_layer, cls_temp_layer, reg_temp_layer_prev, cls_temp_layer_prev):
                                    reg_h, reg_c = reg_temp_block[0]
                                    h_grad = reg_h.grad
                                    c_grad = reg_c.grad
                                    reg_h_prev, reg_c_prev = reg_temp_block_prev[0]
                                    #print("reg_h_prev shape: ", reg_h_prev.shape, reg_h_prev.device)
                                    reg_h_prev.backward(h_grad, retain_graph=retain_graph)
                                    #reg_c_prev.backward(c_grad, retain_graph=retain_graph)

                                    cls_h, cls_c = cls_temp_block[0]
                                    h_grad = cls_h.grad
                                    c_grad = cls_c.grad
                                    cls_h_prev, cls_c_prev = cls_temp_block_prev[0]
                                    cls_h_prev.backward(h_grad, retain_graph=retain_graph)
                                    #cls_c_prev.backward(c_grad, retain_graph=retain_graph)

                    print("Running optimizer step")
                    optimizer.step()
                    scheduler.step()

                    todo_forward_passes= len(states) - 1

                    #states.clear()
                    for i in range(todo_forward_passes):

                        # Repeat forward step after optimized step (to make computation graph of new states consistent with model weights)
                        #if args.truncate_bptt and len(states) > 2:


                        images, gts, counts = data_store[i]
                        images = images.cuda()
                        gts = gts.cuda()
                        counts = counts.cuda()



                        if i == 0 and len(states) < k2:
                            current_reg_hidden_states = None
                            current_cls_hidden_states = None
                        else:
                            current_cls_hidden_states, current_reg_hidden_states = detach_ith_states(states, i)

                        _, _, _, new_reg_hidden_states, new_cls_hidden_states, full_init_reg_hidden_states, full_init_cls_hidden_states = net(
                            images, gts, counts, reg_hidden_states=current_reg_hidden_states,
                            cls_hidden_states=current_cls_hidden_states)


                        if i == 0 and len(states) < k2:
                            states = [None] * len(states)
                            init_states, detached_states = get_init_and_detached_states(full_init_cls_hidden_states,
                                                                                        full_init_reg_hidden_states)

                            states[0] = (None, (init_states[0], init_states[1]))
                            states[1] = ((detached_states[0], detached_states[1]), (new_reg_hidden_states, new_cls_hidden_states))
                        else:
                            states[i+1] = ((current_reg_hidden_states, current_cls_hidden_states),
                                           (new_reg_hidden_states, new_cls_hidden_states))

                    step_counter += 1
                    if should_reset_states(step_counter, grad_accumulate_iterations, train_set_idx, args):
                        print("Resetting states")
                        train_set_idx += 1
                        step_counter = 0
                        states.clear()
                        current_reg_hidden_states = None
                        current_cls_hidden_states = None


                else:
                    print("Running optimizer step")
                    optimizer.step()
                    scheduler.step()

                    torch.cuda.synchronize()

                    step_counter += 1
                    if args.enable_variable_grad_accumulation:
                        train_set_idx += 1
                        grad_accumulate_iterations = args.train_set_sizes[train_set_idx % len(args.train_set_sizes)]
                        states = []
                        current_reg_hidden_states = None
                        current_cls_hidden_states = None


                    elif step_counter * get_data_loader_batch_size(args) * grad_accumulate_iterations >= args.train_set_sizes[train_set_idx % len(args.train_set_sizes)]:
                        train_set_idx += 1
                        step_counter = 0
                        states.clear()
                        current_reg_hidden_states = None
                        current_cls_hidden_states = None

                    elif args.reset_hidden_every_step:
                        states = []
                        current_reg_hidden_states = None
                        current_cls_hidden_states = None

                accumulation_counter = 0
                loss = torch.zeros(1, requires_grad=True).cuda()
                #loss.detach()
                #reg_hidden_states, cls_hidden_states = new_reg_hidden_states, new_cls_hidden_states
            else:
                #step_counter += 1
                scheduler.step()

            #loss.backward()
            #optimizer.step()
            #scheduler.step()

#            pdb.set_trace()
            loc_loss = loss_l.item()
            conf_loss = loss_c.item()
            phase_loss = loss_p.item() if loss_p is not None else 0.0

            
            if loc_loss>300:
                lline = '\n\n\n We got faulty LOCATION loss {} {} {} \n\n\n'.format(loc_loss, conf_loss, phase_loss)
                log_file.write(lline)
                print(lline)
                loc_loss = 20.0
            if conf_loss>300:
                lline = '\n\n\n We got faulty CLASSIFICATION loss {} {} {} \n\n\n'.format(loc_loss, conf_loss, phase_loss)
                log_file.write(lline)
                print(lline)
                conf_loss = 20.0
            if phase_loss>300:
                lline = '\n\n\n We got faulty PHASE loss {} {} {} \n\n\n'.format(loc_loss, conf_loss, phase_loss)
                log_file.write(lline)
                print(lline)
                conf_loss = 20.0
            
            # print('Loss data type ',type(loc_loss))
            loc_losses.update(loc_loss)
            cls_losses.update(conf_loss)
            phase_losses.update(phase_loss)

            avg_loss = (loc_loss + conf_loss)/2.0 if not args.predict_surgical_phase else (loc_loss + conf_loss + phase_loss)/3.0
            losses.update(avg_loss)

            torch.cuda.synchronize()
            batch_time.update(time.perf_counter() - start)
            start = time.perf_counter()

            if iteration % args.log_step == 0 and iteration > args.log_start:
                if args.tensorboard:
                    sw.add_scalars('Classification', {'val': cls_losses.val, 'avg':cls_losses.avg},iteration)
                    sw.add_scalars('Localisation', {'val': loc_losses.val, 'avg':loc_losses.avg},iteration)
                    if args.predict_surgical_phase:
                        sw.add_scalars('Phase', {'val': phase_losses.val, 'avg': phase_losses.avg}, iteration)
                    sw.add_scalars('Overall', {'val': losses.val, 'avg':losses.avg},iteration)

                if args.predict_surgical_phase:
                    print_line = 'Itration [{:d}]{:06d}/{:06d} loc-loss {:.2f}({:.2f}) cls-loss {:.2f}({:.2f}) phase-loss {:.2f}({:.2f}) ' \
                                 'average-loss {:.2f}({:.2f}) DataTime{:0.2f}({:0.2f}) Timer {:0.2f}({:0.2f})'.format(
                        epoch,
                        iteration, args.max_iter, loc_losses.val, loc_losses.avg, cls_losses.val,
                        cls_losses.avg, phase_losses.val, phase_losses.avg, losses.val, losses.avg, 10 * data_time.val, 10 * data_time.avg,
                        10 * batch_time.val, 10 * batch_time.avg)
                else:
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
                    sw.add_scalar('mAP@0.3', mAP, iteration)
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


def detach_ith_states(states, i=-1):
    reg_hidden_states, cls_hidden_states = states[i][1]
    current_reg_hidden_states, current_cls_hidden_states = [], []
    for reg_hidden, cls_hidden in zip(reg_hidden_states, cls_hidden_states):
        current_reg_fpn_ftr, current_cls_fpn_ftr = [], []
        for reg_temp_layer, cls_temp_layer in zip(reg_hidden, cls_hidden):
            current_reg_temp_layer, current_cls_temp_layer = [], []
            for reg_temp_block, cls_temp_block in zip(reg_temp_layer, cls_temp_layer):

                current_reg_temp_block, current_cls_temp_block = [], []
                reg_h, reg_c = reg_temp_block[0]  # Hard coded for now to assume single layer convlstm block
                detach_reg_h = reg_h.detach()
                detach_reg_c = reg_c.detach()
                detach_reg_h.requires_grad = True
                detach_reg_c.requires_grad = True

                current_reg_temp_block.append((detach_reg_h, detach_reg_c))

                cls_h, cls_c = cls_temp_block[0]
                detach_cls_h = cls_h.detach()
                detach_cls_c = cls_c.detach()
                detach_cls_h.requires_grad = True
                detach_cls_c.requires_grad = True

                current_cls_temp_block.append((detach_cls_h, detach_cls_c))
                current_reg_temp_layer.append(current_reg_temp_block)
                current_cls_temp_layer.append(current_cls_temp_block)
            current_reg_fpn_ftr.append(current_reg_temp_layer)
            current_cls_fpn_ftr.append(current_cls_temp_layer)

        current_reg_hidden_states.append(current_reg_fpn_ftr)
        current_cls_hidden_states.append(current_cls_fpn_ftr)
    return current_cls_hidden_states, current_reg_hidden_states


def get_init_and_detached_states(full_init_cls_hidden_states, full_init_reg_hidden_states):
    init_reg_hidden_states, init_cls_hidden_states = [], []
    detached_reg_hidden_states, detached_cls_hidden_states = [], []
    for reg_hidden, cls_hidden in zip(full_init_reg_hidden_states, full_init_cls_hidden_states):
        init_reg_fpn_ftr, init_cls_fpn_ftr = [], []
        detached_reg_fpn_ftr, detached_cls_fpn_ftr = [], []
        for reg_temp_layer, cls_temp_layer in zip(reg_hidden, cls_hidden):
            init_reg_temp_blocks, init_cls_temp_blocks = [], []
            detached_reg_temp_blocks, detached_cls_temp_blocks = [], []
            for reg_temp_block, cls_temp_block in zip(reg_temp_layer, cls_temp_layer):
                init_reg_temp_blocks.append(reg_temp_block[0])
                detached_reg_temp_blocks.append(reg_temp_block[1])

                init_cls_temp_blocks.append(cls_temp_block[0])
                detached_cls_temp_blocks.append(cls_temp_block[1])

            init_reg_fpn_ftr.append(init_reg_temp_blocks)
            init_cls_fpn_ftr.append(init_cls_temp_blocks)

            detached_reg_fpn_ftr.append(detached_reg_temp_blocks)
            detached_cls_fpn_ftr.append(detached_cls_temp_blocks)

        init_reg_hidden_states.append(init_reg_fpn_ftr)
        init_cls_hidden_states.append(init_cls_fpn_ftr)

        detached_reg_hidden_states.append(detached_reg_fpn_ftr)
        detached_cls_hidden_states.append(detached_cls_fpn_ftr)
    return (init_reg_hidden_states, init_cls_hidden_states), (detached_reg_hidden_states, detached_cls_hidden_states)


def init_states_list():
    return [], []


def get_data_loader_batch_size(args):
    #return args.batch_size if not args.time_distributed_backbone else args.batch_size + args.temporal_slice_timesteps - 1
    return args.batch_size if not args.time_distributed_backbone else args.batch_size * args.temporal_slice_timesteps

def should_backprop(accumulation_counter, grad_accumulate_iterations, step_counter, train_set_idx, args):
    if accumulation_counter % grad_accumulate_iterations ==0:
        return True
    elif args.truncate_bptt and args.override_optimization_steps_tbptt and step_counter % args.overrided_optimization_steps == 0:
        return True
    elif args.truncate_bptt and (accumulation_counter*get_data_loader_batch_size(args) + (step_counter * get_data_loader_batch_size(args) * grad_accumulate_iterations)) >= args.train_set_sizes[train_set_idx % len(args.train_set_sizes)]:
        return True
    return False
def should_reset_states(step_counter, grad_accumulate_iterations, train_set_idx, args):
    if step_counter * get_data_loader_batch_size(args) * grad_accumulate_iterations >= args.train_set_sizes[train_set_idx % len(args.train_set_sizes)]:
        return True
    if args.override_optimization_steps_tbptt and step_counter % args.overrided_optimization_steps == 0:
        return True
    return False
def construct_temporal_batches(images, batch_size, timesteps):
    _, channels, height, width = images.shape
    #images_td = torch.tensor([])
    #for seq_idx in range(batch_size):
    #    seq = images[seq_idx:seq_idx + timesteps, :,:,:]
    #    images_td = torch.cat((images_td, torch.unsqueeze(seq, 0)), 0)
    #return images_td
    return images.view(batch_size, timesteps, channels, height, width)

def load_model_state_dict(model_file_name, net):
    net_state_dict = torch.load(model_file_name)
    #net_state_dict = {k.replace("module.",""): net_state_dict[k] for k in net_state_dict}
    #for k in list(net_state_dict.keys()): #Renames backbone_net keys to backbone_net.td_net for time distributed network
    #    if "backbone_net" in k:
    #        net_state_dict[k.replace("backbone_net.","backbone_net.td_net.")] = net_state_dict[k]
    #        del net_state_dict[k]
    #torch.save(net_state_dict, '{:s}/model_{:06d}.pth'.format(args.save_root, 14000))
    net.load_state_dict(net_state_dict, strict=False)


def param_dict_exists(param_dict, param_dicts):
    for dest_param_dict in param_dicts:
        if dest_param_dict['name'] in param_dict['name']:
            return True
    return False

def load_optim_state_dict(optimizer, optimizer_file_name):
    optim_state_dict = torch.load(optimizer_file_name)
    loaded_state_dict = optim_state_dict['param_groups']
    #Filter param_groups list
    param_dicts_map = {idx: param_dict for idx, param_dict in enumerate(loaded_state_dict) if param_dict_exists(param_dict, optimizer.state_dict()['param_groups'])}
    optim_state_dict['param_groups'] = list(param_dicts_map.values())
    optim_state_dict['state'] = {k: optim_state_dict['state'][k] for k in param_dicts_map}
    optimizer.load_state_dict(optim_state_dict)

def cat_timestep(tensor, tensor_td_batch):
    return torch.cat((tensor_td_batch, torch.unsqueeze(tensor, 0)), 0)


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
    reg_hidden_states = None
    cls_hidden_states = None
    with torch.no_grad():
        for val_itr, (images, targets, batch_counts, img_indexs, wh) in enumerate(val_data_loader):

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_size = images.size(0)

            if args.time_distributed_backbone:
                _, channels, height, width = images.shape
                images = construct_temporal_batches(images, args.batch_size, args.temporal_slice_timesteps)
                #targets, counts = generate_temporal_gts(targets, batch_counts, args.batch_size, args.temporal_slice_timesteps)

            images = images.cuda(0, non_blocking=True)



            decoded_boxes, conf_data, reg_hidden_states, cls_hidden_states,_, _ = net(images, reg_hidden_states=reg_hidden_states, cls_hidden_states=cls_hidden_states)

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
