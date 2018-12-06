import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable


import os
#print(os.listdir('./'))

import sys
sys.path.insert(0, './../')
from experiment_real.dataset import Dictionary, VQAFeatureDataset
import model_features as base_model


os.chdir('./../')


exp = './experiment_real/'


torch.manual_seed(1111)
torch.cuda.manual_seed(1111)
torch.backends.cudnn.benchmark = True


dictionary = Dictionary.load_from_file(exp+'data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary, dataroot=exp+'data/')
eval_dset = VQAFeatureDataset('val', dictionary, dataroot=exp+'data/')
batch_size = 512
num_hid = 1024

constructor = 'build_baseline0_newatt'
model = getattr(base_model, constructor)(train_dset, num_hid).cuda()
model.w_emb.init_embedding(exp+'data/glove6b_init_300d.npy')

train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)



model = nn.DataParallel(model).cuda()

model_path = exp+'saved_models/my_model_1/model.pth'
model_params = torch.load(model_path)
model.load_state_dict(model_params, strict=False)
model.train(False)
model.eval()
pass


def train(model, data_loader):
    with open('./visualization/data/responses_real_val', 'w') as file:
        for i, (v, b, q, a, q_id) in enumerate(data_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            pred = model(v, b, q, a)
            resps = pred.cpu().detach().numpy()
            
            query_ids = q_id.cpu().detach().numpy()
            
            for j, res in enumerate(resps):
                value = '\t'.join(res.astype(str).tolist())+'\t'+str(query_ids[j])
                file.write(value)
                file.write('\n')
    print('END features')

responses = train(model, eval_loader)









