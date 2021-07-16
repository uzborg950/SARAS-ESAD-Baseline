from .resnetFPN import resnetfpn
import torch
import pdb
from models.time_distributed import TimeDistributed5D
def backbone_models(modelname, model_dir, use_bias, args):

    if modelname[:6] == 'resnet':
        # print(modelname)
        modelperms = {'resnet18': [2, 2, 2, 2], 'resnet34': [3, 4, 6, 3], 'resnet50': [3, 4, 6, 3],
                      'resnet101': [3, 4, 23, 3], 'resnet152': [3, 8, 36, 3]}
        model = resnetfpn(modelperms[modelname], modelname, use_bias)
        args.predictor_layers = [3,4,5,6,7]
        # print('here here', model)
        if len(model_dir)>1: # load imagenet pretrained weights
            load_dict = torch.load(model_dir + modelname+'.pth')
            model.load_my_state_dict(load_dict)
        if args.time_distributed_backbone:
            model = TimeDistributed5D(model, args.temporal_slice_timesteps, fpn = True)
        return model
