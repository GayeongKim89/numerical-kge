import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.autograd import Variable

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

def get_uni_param(shape):
    param = nn.Parameter(torch.Tensor(shape))
    # nn.init.xavier_uniform_(param.data)
    return param

class Gate(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):

        super(Gate, self).__init__()
        self.output_size = output_size

        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size-output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit],x_lit.ndimension()-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output






class RAKGEModel(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(RAKGEModel, self).__init__()

        self.bceloss = torch.nn.BCELoss()


        self.p = params
        self.init_embed = get_param((num_ents, self.p.init_dim))
        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num
        self.num_ents = num_ents

        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.att_dim)
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)

        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))
        self.numerical_literals = Variable(torch.from_numpy(numerical_literals)).cuda()


        self.init_rel = get_param((num_rels , self.p.init_dim))

        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.inp_drop = nn.Dropout(p=self.p.input_drop)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_tail = nn.CosineSimilarity(dim=2, eps=1e-6)
        self.alpha = get_param((num_rels, 1))
        self.scale = get_uni_param(1)
        self.s = get_uni_param((self.num_ents, 1))



        self.h_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })


        self.t_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.num_ents, self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.r_ops_dict = nn.ModuleDict({
            'p': nn.Linear(self.p.init_dim, self.p.gcn_dim, bias=False),
            'b': nn.BatchNorm1d(self.p.gcn_dim),
            'd': nn.Dropout(self.p.hid_drop),
            'a': nn.Tanh(),
        })

        self.x_ops = self.p.x_ops
        self.r_ops = self.p.r_ops
        self.diff_ht = False


    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def exop(self, x, r, x_ops=None, r_ops=None, diff_ht=False):
        x_head = x_tail = x
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r


    def exop_last(self, x_head, x_tail, r, x_ops=None, r_ops=None, diff_ht=True):
        if len(x_ops) > 0:
            for x_op in x_ops.split("."):
                if diff_ht:
                    #print('sss')
                    x_head = self.h_ops_dict[x_op](x_head)
                    x_tail = self.t_ops_dict[x_op](x_tail)
                else:
                    x_head = x_tail = self.h_ops_dict[x_op](x_head)

        if len(r_ops) > 0:
            for r_op in r_ops.split("."):
                r = self.r_ops_dict[r_op](r)

        return x_head, x_tail, r






class TransE_Gate_att(RAKGEModel):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, numerical_literals, params)
        self.loop_emb = get_param([1, self.p.init_dim])



    def forward(self, g, sub, rel, obj, e2_multi, pos_neg):
        x_h = self.init_embed-self.loop_emb
        x_t = self.init_embed-self.loop_emb
        r = self.init_rel
        a = self.alpha

        e1_emb = torch.index_select(x_h, 0, sub) #(batch, init_dim)
        rel_emb = torch.index_select(r, 0, rel) #(batch, init_dim)
        e2_emb = torch.index_select(x_t, 0, obj)



        e2_multi_emb = x_t #(entity_num, init_dim)
        numerical_literals = self.numerical_literals

        # Begin literals
        e1_num_lit = torch.index_select(numerical_literals, 0, sub)  # (batch_size, att_dim)

        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb)).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze(0))  # (batch_num, emb_dim)

        ###### tail
        # begin literals
        numerical_literals = numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)
        e2_multi_num_lit, _ = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                  e2_emb_att)  # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb,  self.x_ops, self.r_ops)



        obj_emb = e1_emb + rel_emb # (batch, dim)


        x = self.p.gamma - \
            torch.norm(obj_emb.unsqueeze(1) - e2_multi_emb, p=1, dim=2)
        score = torch.sigmoid(x)






        return score, loss