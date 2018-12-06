import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from attention import Attention, NewAttention
from fc import FCNet
import os

import sys
sys.path.insert(0, './../')
from language_model import WordEmbedding, QuestionEmbedding



class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, u_net, classifier):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.u_net = u_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim] 
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q) # (A) words2vec
        q_emb = self.q_emb(w_emb) # (B) [batch, q_dim] questions2vec 

        att = self.v_att(v, q_emb) # (C) improved attention
        v_emb = (att * v).sum(1) # (D) [batch, v_dim]

        u_emb = torch.mean(v, 1) # ( ) [batch, v_dim] [TODO TODO]
        
        q_repr = self.q_net(q_emb) # (E) match size - QUESTION 
        v_repr_td = self.v_net(v_emb) # (F) match size - IMG TOP DOWN
        v_repr_bu = self.u_net(u_emb) # ( ) match size - IMG BOTTOM UP 
        
        joint_repr_td = q_repr * v_repr_td # (G)
        joint_repr_bu = q_repr * v_repr_bu # ( )

        joint_repr = torch.cat((joint_repr_td, joint_repr_bu), 1)
        
        logits = self.classifier(joint_repr) # (H)
        return logits


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout): # out_dim = responses
        super(SimpleClassifier, self).__init__()
        layers = [
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
        ]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.main(x)
        return logits
    
    
def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    
    q_net = FCNet([q_emb.num_hid, num_hid]) # match dimensions
    v_net = FCNet([dataset.v_dim, num_hid]) # match dimensions
    u_net = FCNet([dataset.v_dim, num_hid]) # match dimensions

    classifier = SimpleClassifier(
        num_hid *2 , num_hid * 2, dataset.num_ans_candidates, 0.5)
    
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, u_net, classifier)

