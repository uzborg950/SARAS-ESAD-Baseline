import torch, pdb
import torch.optim as optim
# from .madamw import Adam as AdamM
# from .adamw import Adam as AdamW
# from torch.optim.lr_scheduler import MultiStepLR

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        self.milestones = milestones
        self.gammas = gammas
        self.verbose = False
        assert len(gammas) == len(milestones), 'Milestones and gammas should be of same length gammas are of len ' + (len(gammas)) + ' and milestones '+ str(len(milestones))
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)#(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            index = self.milestones.index(self.last_epoch)
            return [group['lr'] * self.gammas[index] for group in self.optimizer.param_groups]
    def reduce_lr(self):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] /100
    def reset_lr(self, lr):
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    
    def print_lr(self, is_verbose, group, lr, epoch=None): #The extra arguments have been added to deal with the new inheritted method in lr_scheduler
        pass
        #print([[group['name'], group['lr']] for group in self.optimizer.param_groups])

def get_optim(args, net):
    freeze_layers = ['backbone_net.layer'+str(n) for n in range(1, args.freezeupto+1)]
    params = []
    solver_print_str = '\n\nSolver configs are as follow \n\n\n'


    for key, value in net.named_parameters():

        if args.freezeupto>0 and (key.find('backbone_net.conv1')>-1 or key.find('backbone_net.bn1')>-1): # Freeze first conv layer and bn layer in resnet
            value.requires_grad = False
            continue
        if key.find('cls_heads') > -1:
            if (args.freeze_cls_heads):
                value.requires_grad = False
                continue
        elif key.find('reg_heads') > -1:
            if (args.freeze_reg_heads):
                value.requires_grad = False
                continue
        elif key.find('backbone_net')>-1:
            if (args.freeze_backbone):
                value.requires_grad = False
                continue
            else:
                for layer_id in freeze_layers:
                    if key.find(layer_id)>-1:
                        value.requires_grad = False
                        continue

        if not value.requires_grad:
            continue

        lr = args.lr
        wd = args.weight_decay

        if args.optim == 'ADAM':
            wd = 0.0

        if "bias" in key:
            lr = lr*2.0

        if args.optim == 'SGD':
            params += [{"params": [value], "name":key, "lr": lr, "weight_decay":wd, "momentum":args.momentum}]
        else:
            params += [{"params": [value], "name":key, "lr": lr, "weight_decay":wd}]

        print_l = key +' is trained at the rate of ' + str(lr)
        print(print_l)
        solver_print_str += print_l + '\n'
        
        
    if args.optim == 'SGD':
        optimizer = optim.SGD(params)
    elif args.optim == 'ADAM':
        optimizer = optim.Adam(params)
    # elif args.optim == 'ADAMW':
    #     optimizer = AdamW(params)
    # elif args.optim == 'ADAMM':
    #     optimizer = AdamM(params)
    else:
        error('Define optimiser type ')
    
    solver_print_str += 'optimizer is '+ args.optim + '\nDone solver configs\n\n'
    
    scheduler = WarmupMultiStepLR(optimizer, args.milestones, args.gammas)

    return optimizer, scheduler, solver_print_str