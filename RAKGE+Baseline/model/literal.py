import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np
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
        x = torch.cat([x_ent, x_lit], x_lit.ndimension()-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1-gate) * x_ent + gate * g_embedded

        return output


def our_loss(e1_emb, rel_emb, e2_multi, e2_multi_emb):
    prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
    e2_multi_prob = e2_multi * prob

    C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
    positive_mean = C.sum(dim=1) / e2_multi_prob.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)

    alpha, beta = np.random.uniform(0, 1.0), np.random.uniform(0, 1.0)
    positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb

    # # # negative meadn
    e2_multi_inv = 1 - e2_multi
    e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)

    N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
    negative_mean = N.sum(dim=1) / e2_multi_inv.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)

    negative_mean = beta * negative_mean + (1 - beta) * e1_emb

    pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1,keepdim=True)
    neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)


    loss = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))


    return loss

class DistMultLiteral_gate(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(DistMultLiteral_gate, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.emb_dim))
        self.emb_rel = get_param((num_rels * 2, self.emb_dim))

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        # numerical_literals = torch.from_numpy(numerical_literals).cuda()
        # self.numerical_literals = torch.where(numerical_literals > 0, 1.0, 0.0)
        self.num_att = numerical_literals.shape[1]

        self.emb_num_lit = Gate(self.emb_dim + self.num_att, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)


    def forward(self, g, e1, rel, e2_multi):

        e1_emb = torch.index_select(self.emb_e, 0, e1)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)
        e2_multi_emb = self.emb_e

        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)
        e2_num_lit = self.numerical_literals

        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(e2_multi_emb, e2_num_lit)

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        obj_emb = e1_emb * rel_emb
        pred = torch.mm(obj_emb, e2_multi_emb.transpose(1, 0))
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)

        return pred, 0

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class TransELiteral_gate(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(TransELiteral_gate, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels * 2, self.p.init_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        self.n_num_lit = self.numerical_literals.size(1)

        # gating
        self.emb_num_lit = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

    def forward(self, g, e1, rel, e2_multi):
        e1_emb = torch.index_select(self.emb_e, 0, e1)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)
        e2_multi_emb = self.emb_e

        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)
        e2_num_lit = self.numerical_literals


        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(e2_multi_emb, e2_num_lit)

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        loss = our_loss(e1_emb, rel_emb, e2_multi, e2_multi_emb)



        score = self.p.gamma - \
            torch.norm(((e1_emb + rel_emb).unsqueeze(1) - e2_multi_emb.unsqueeze(0)), p=1, dim=2)

        pred = torch.sigmoid(score)


        return pred, loss

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class ComplExLiteral_gate(torch.nn.Module):

    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(ComplExLiteral_gate, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim

        # entity, relation embedding table
        self.emb_e_real = get_param((num_ents, self.emb_dim))
        self.emb_e_img = get_param((num_ents, self.emb_dim))
        self.emb_rel_real = get_param((num_rels * 2, self.emb_dim))
        self.emb_rel_img = get_param((num_rels * 2, self.emb_dim))

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        numerical_literals = torch.from_numpy(numerical_literals).cuda()
        self.numerical_literals = torch.where(numerical_literals > 0, 1.0, 0.0)
        self.num_att = numerical_literals.shape[1]

        self.emb_num_lit_real = Gate(self.emb_dim + self.num_att, self.emb_dim)
        self.emb_num_lit_img = Gate(self.emb_dim + self.num_att, self.emb_dim)

        # Dropout + loss
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)


    def forward(self,g, e1, rel, e2_multi):
        e1_emb_real = torch.index_select(self.emb_e_real, 0, e1)
        rel_emb_real = torch.index_select(self.emb_rel_real, 0, rel)

        e1_emb_img = torch.index_select(self.emb_e_img, 0, e1)
        rel_emb_img = torch.index_select(self.emb_rel_img, 0, e1)

        e2_multi_emb_real=self.emb_e_real
        e2_multi_emb_img=self.emb_e_img


        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)
        e1_emb_real = self.emb_num_lit_real(e1_emb_real, e1_num_lit)
        e1_emb_img = self.emb_num_lit_img(e1_emb_img, e1_num_lit)

        e2_multi_emb_real = self.emb_num_lit_real(e2_multi_emb_real, self.numerical_literals)
        e2_multi_emb_img = self.emb_num_lit_img(e2_multi_emb_img, self.numerical_literals)

        # End literals
        e1_emb_real = self.inp_drop(e1_emb_real)
        rel_emb_real = self.inp_drop(rel_emb_real)
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_img = self.inp_drop(rel_emb_img)

        realrealreal = torch.mm(e1_emb_real * rel_emb_real, e2_multi_emb_real.t())
        realimgimg = torch.mm(e1_emb_real * rel_emb_img, e2_multi_emb_img.t())
        imgrealimg = torch.mm(e1_emb_img * rel_emb_real, e2_multi_emb_img.t())
        imgimgreal = torch.mm(e1_emb_img * rel_emb_img, e2_multi_emb_real.t())

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)

        return pred, 0

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class ConvELiteral_gate(torch.nn.Module):

    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(ConvELiteral_gate, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.emb_dim))
        self.emb_rel = get_param((num_rels * 2, self.emb_dim))

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        self.num_att = numerical_literals.shape[1]

        self.emb_num_lit = Gate(self.emb_dim + self.num_att, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)
        self.hidden_drop = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.feat_drop)

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=self.p.bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(self.emb_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_ents)))
        self.fc = torch.nn.Linear(10368, self.emb_dim)


    def forward(self, g, e1, rel, e2_multi):
        e1_emb = torch.index_select(self.emb_e, 0, e1)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)

        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit)
        e2_multi_emb = self.emb_num_lit(self.emb_e, self.numerical_literals)

        # End literals
        e1_emb = e1_emb.view(len(e1), 1, 10, self.emb_dim//10)
        rel_emb = rel_emb.view(len(e1), 1, 10, self.emb_dim//10)

        stacked_inputs = torch.cat([e1_emb, rel_emb], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x = self.inp_drop(stacked_inputs)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(len(e1), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, e2_multi_emb.t())
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred, 0

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class KBLN(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, c, var, params=None):
        super(KBLN, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params
        self.num_ents = num_ents

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels * 2, self.p.init_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.c = Variable(torch.FloatTensor(c)).cuda()
        self.var = Variable(torch.FloatTensor(var)).cuda()

        self.nf_weights = get_param((num_rels *2, self.n_num_lit))

        # gating
        self.emb_num_lit = Gate(self.emb_dim+self.n_num_lit, self.emb_dim)
        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.hid_drop)

    def forward(self, g, e1, rel, e2_multi):
        e1_emb = torch.index_select(self.emb_e, 0, e1)
        #print('e1',e1.shape)
        #print(e1_emb.shape)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)

        e1_emb = self.inp_drop(e1_emb)
        rel_emb = self.inp_drop(rel_emb)

        e2_multi_emb = self.emb_e

        score_l = torch.mm(e1_emb * rel_emb, e2_multi_emb.transpose(1,0))

        n_h = torch.index_select(self.numerical_literals, 0, e1) #(batch, n_lits)

        #print(n_h.shape)
        n_t = self.numerical_literals #(n_ents, n_lits)



        # Features (batch_size x num_ents x n_lit)
        n = n_h.unsqueeze(1).repeat(1, self.num_ents, 1) - n_t
        phi = self.rbf(n) #(batch, num_ents, n_lits)

        # Weights (batch_size, 1, n_lits)
        w_nf = torch.index_select(self.nf_weights, 0, rel)

        # (batch_size, num_ents)
        score_n = torch.bmm(phi, w_nf.unsqueeze(2)).squeeze(2)
        """ End numerical literals """

        score = F.sigmoid(score_l + score_n)

        return score, 0

    def rbf(self, n):
        """
        Apply RBF kernel parameterized by (fixed) c and var, pointwise.
        n: (batch_size, num_ents, n_lit)
        """
        return torch.exp(-(n - self.c)**2 / self.var)

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


