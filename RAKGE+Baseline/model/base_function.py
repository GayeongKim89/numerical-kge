import torch
from torch import nn


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
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


class Literal_Our_Model(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(Literal_Our_Model, self).__init__()

        self.bceloss = torch.nn.BCELoss()

        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels * 2, self.p.init_dim))

        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        # relation projection
        self.linear=nn.Linear(self.p.init_dim,self.att_dim)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)
        self.multihead_attn2 = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.emb_num_lit2 = Gate(self.att_dim + self.emb_dim, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)


    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class LiteralModel(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(LiteralModel, self).__init__()

        self.bceloss = torch.nn.BCELoss()

        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels * 2, self.p.init_dim))

        # attribute
        self.num_att = numerical_literals.shape[1]
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        # gating
        self.emb_num_lit = Gate(self.num_att + self.emb_dim, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)




