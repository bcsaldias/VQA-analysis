import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


"""
CHANGE MODEL HERE
"""
from experiment_real.dataset import Dictionary, VQAFeatureDataset
from models import base_model

from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--num_hid', type=int, default=1024)
    parser.add_argument('--model', type=str, default='baseline0_newatt')
    parser.add_argument('--output', type=str, default='saved_models/exp0')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    dictionary = Dictionary.load_from_file('experiment_real/data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, dataroot='experiment_real/data/')
    eval_dset = VQAFeatureDataset('val', dictionary, dataroot='experiment_real/data/')
    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args.num_hid).cuda()
    model.w_emb.init_embedding('experiment_real/data/glove6b_init_300d.npy')
    


    train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
    eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)
    
    model = nn.DataParallel(model).cuda()

    
    """
    load pretrained model
    model_path = 'saved_models/my_6_abstract/model.pth'
    model_params = torch.load(model_path)
    print("# params", len(model_params.keys()))
    model.load_state_dict(model_params, strict=False)
    print("Loading params")
    model.eval() 
    model.train()
    """

    print("PASE")
    
    train(model, train_loader, eval_loader, args.epochs, args.output)

