import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


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

        
    ########################
    # features, spatials, question, target !!! should receive new features!!
    ########################
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

        #print("sum", type(v.sum(1)))
        #print("max", type(torch.max(v, 1)))
        #print("median", type(torch.median(v, 1)))
        u_emb, indexes = torch.median(v, 1) # ( ) [batch, v_dim] [TODO TODO]
        
        q_repr = self.q_net(q_emb) # (E) match size - QUESTION 
        v_repr_td = self.v_net(v_emb) # (F) match size - IMG TOP DOWN
        v_repr_bu = self.u_net(u_emb) # ( ) match size - IMG BOTTOM UP 
        
        joint_repr_td = q_repr * v_repr_td # (G)
        joint_repr_bu = q_repr * v_repr_bu # ( )

        joint_repr = torch.cat((joint_repr_td, joint_repr_bu), 1)
        
        logits = self.classifier(joint_repr) # (H)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    u_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, u_net, classifier)


######################## 
# modify to add new features
######################## 
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
