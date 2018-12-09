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
import model_predict as base_model


os.chdir('./../')


exp = './experiment_abstract/'

torch.manual_seed(1111)
torch.cuda.manual_seed(1111)
torch.backends.cudnn.benchmark = True

dictionary = Dictionary.load_from_file(exp+'data/dictionary.pkl')
train_dset = VQAFeatureDataset('train', dictionary, dataroot=exp+'data/')
eval_dset = VQAFeatureDataset('val', dictionary, dataroot=exp+'data/')
batch_size = 1
num_hid = 1024

constructor = 'build_baseline0_newatt'
model = getattr(base_model, constructor)(train_dset, num_hid).cuda()
model.w_emb.init_embedding(exp+'data/glove6b_init_300d.npy')

train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=1)
eval_loader =  DataLoader(eval_dset, batch_size, shuffle=True, num_workers=1)

model = nn.DataParallel(model).cuda()


model_path = exp+'saved_models/my_model_2/model.pth'
model_params = torch.load(model_path)
model.load_state_dict(model_params)
model.eval()
pass



def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def save_grad(model, dataloader, num_epochs, output):
        
    with open('./analysis/data/gradients_{}'.format(output), 'w') as file:
        
        optim = torch.optim.Adamax(model.parameters())

        for epoch in range(num_epochs):
            model.train()

            for i, OUT in enumerate(dataloader):
                (v, b, q, a, q_id) = OUT
                v = Variable(v).cuda()
                b = Variable(b).cuda()
                q = Variable(q).cuda()
                a = Variable(a).cuda()

                query_ids = q_id.cpu().detach().numpy()
                assert len(query_ids) == 1
                
                pred = model(v, b, q, a)
                loss = instance_bce_with_logits(pred, a)
                loss.backward()
                nn.utils.clip_grad_norm(model.parameters(), 0.25)

                instance_grad = optim.param_groups[0]['params'][-4].sum(1).cpu().detach()
                instance_grad = instance_grad.numpy().astype(str).tolist()
                
                file.write(','.join(instance_grad)+','+str(query_ids[0]))
                file.write('\n')

                optim.zero_grad()



responses = save_grad(model, eval_loader, 1, 'abstract_val')
print('END features')
responses = save_grad(model, train_loader, 1, 'abstract_train')
print('END features')









