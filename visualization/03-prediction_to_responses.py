import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable
import numpy as np

import os
#print(os.listdir('./'))

import sys
sys.path.insert(0, './../')
from experiment_abstract.dataset import Dictionary, VQAFeatureDataset
import base_model_predict as base_model


os.chdir('./../')


exp = './experiment_abstract/'

dictionary = Dictionary.load_from_file(exp+'data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary, dataroot=exp+'data/')
#eval_dset = VQAFeatureDataset('val', dictionary, dataroot=exp+'data/')
batch_size = 512
num_hid = 1024

constructor = 'build_baseline0_newatt'
model = getattr(base_model, constructor)(train_dset, num_hid).cuda()
model.w_emb.init_embedding(exp+'data/glove6b_init_300d.npy')

train_loader = DataLoader(train_dset, batch_size, shuffle=False, num_workers=1)
#eval_loader =  DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1)

model = nn.DataParallel(model).cuda()


model_path = exp+'saved_models/base_model_again_1/model.pth'
model_params = torch.load(model_path)
model.load_state_dict(model_params)
model.train(False)
model.eval()
pass
    
    
def compare(logits, labels):
    logits = torch.max(logits, 1)[1].data
    labels = torch.max(labels, 1)[1].data

    return np.array(logits==labels).astype(int).tolist()
    
def compare(logits, labels):
    logits = torch.max(logits, 1)[1].data.cpu().numpy().astype(str).tolist()
    labels = torch.max(labels, 1)[1].data.cpu().numpy().astype(str).tolist()
    return list(zip(logits, labels))
    
def evaluate(model, dataloader):
    with open('./visualization/data/predictions_details_base_abstract_train', 'w') as file:
        for v, b, q, a in iter(dataloader):
            with torch.no_grad():
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                pred = model(v, b, q, None)
                batch_score = compare(pred, a.cuda())

                for value in batch_score:
                    file.write(','.join(value))
                    file.write('\n')
    print('END features')
    
responses = evaluate(model, train_loader)









