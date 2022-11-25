import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from pathlib import Path
from torch.distributions import Beta
from torch import autograd

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

        self.r = nn.Linear(input_size - output_size, output_size)
        self.r1 = nn.Linear(output_size, output_size, bias=False)
        self.r2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.reset_bias = nn.Parameter(torch.zeros(output_size))
        self.gg = nn.Linear(output_size, output_size, bias=False)
        self.ll = torch.nn.Linear(input_size + input_size, output_size)

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit],x_lit.ndimension()-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)

        output = (1-gate) * x_ent + gate * g_embedded
        #내가 바꿈
        # output = (1-gate) * x_ent + gate * x_lit

        return output

    # 내가 바꿈
    def forward(self, x_ent, x_lit):

        reset = self.gate_activation(self.r1(x_ent)+self.r2(x_lit) + self.reset_bias)
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
    #     # print(self.g(x_ent).shape)
    #     # print(reset.shape)
    #     # print(self.r(x_lit).shape)
        tem = torch.tanh(self.gg(x_ent)*reset +self.r(x_lit))
        output = (1-gate) * x_ent + gate * tem

        return output
    #Linear
    '''def forward(self,x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], x_lit.ndimension() - 1)
        output = self.g(x)
        print(output)
        return output'''

    '''def forward(self,x_ent, x_lit):
        output = x_ent + x_lit
        return output'''

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

        self.r = nn.Linear(input_size - output_size, output_size)
        self.r1 = nn.Linear(output_size, output_size, bias=False)
        self.r2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.reset_bias = nn.Parameter(torch.zeros(output_size))
        self.gg = nn.Linear(output_size, output_size, bias=False)

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit],x_lit.ndimension()-1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)

        output = (1-gate) * x_ent + gate * g_embedded
        #내가 바꿈
        # output = (1-gate) * x_ent + gate * x_lit

        return output

    # def forward(self, x_ent, x_lit):
        # x = torch.cat([x_ent, x_lit],x_lit.ndimension()-1)
        # g_embedded = torch.tanh(self.g(x))
        # gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)

        # output = (1-gate) * x_ent + gate * x_lit
        #내가 바꿈
        # output = (1-gate) * x_ent + gate * x_lit

        # return output
    # 내가 바꿈
    # def forward(self, x_ent, x_lit):
    #
    #     reset = self.gate_activation(self.r1(x_ent)+self.r2(x_lit) + self.reset_bias)
    #     gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
    # #     # print(self.g(x_ent).shape)
    # #     # print(reset.shape)
    # #     # print(self.r(x_lit).shape)
    #     tem = torch.tanh(self.gg(x_ent)*reset +self.r(x_lit))
    #     output = (1-gate) * x_ent + gate * tem
    #
    #     return output




class ComplEx_Gate_att(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(ComplEx_Gate_att, self).__init__()

        self.bceloss = torch.nn.BCELoss()

        self.p = params
        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entiity embedding
        self.emb_e_real = get_param((num_ents, self.emb_dim))
        self.emb_e_img = get_param((num_ents, self.emb_dim))

        # relation embedding
        self.emb_rel_real = get_param((num_rels * 2, self.emb_dim))
        self.emb_rel_img = get_param((num_rels * 2, self.emb_dim))

        # attribute embedding
        self.num_att = numerical_literals.shape[1]
        self.emb_att_real = get_param((self.num_att, self.att_dim))
        self.emb_att_img = get_param((self.num_att, self.att_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        # relation projection
        self.linear_real = nn.Linear(self.emb_dim, self.att_dim)
        self.linear_img = nn.Linear(self.emb_dim, self.att_dim)

        # MultiheadAttention
        self.multihead_attn_real = torch.nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)
        self.multihead_attn_img = torch.nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit_real = Gate(self.emb_dim + self.att_dim, self.emb_dim)
        self.emb_num_lit_img = Gate(self.emb_dim + self.att_dim, self.emb_dim)

        self.emb_num_lit_real2 = Gate(self.emb_dim + self.att_dim, self.emb_dim)
        self.emb_num_lit_img2 = Gate(self.emb_dim + self.att_dim, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)


    def forward(self, g, e1, rel, e2_multi):
        e1_emb_real = torch.index_select(self.emb_e_real, 0, e1)  # (batch_size, emb_dim)
        rel_emb_real = torch.index_select(self.emb_rel_real, 0, rel)  # (batch_size, emb_dim)
        e2_multi_emb_real = self.emb_e_real

        e1_emb_img = torch.index_select(self.emb_e_img, 0, e1)
        rel_emb_img = torch.index_select(self.emb_rel_img, 0, rel)
        e2_multi_emb_img = self.emb_e_img


        ###### head real part
        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)  # (batch_size, emb_dim)
        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att_real = self.emb_att_real.unsqueeze(1)  # (att_num, 1, emb_dim)
        e1_emb_att_real = e1_num_lit * emb_att_real  # (att_num, batch_num, emb_dim)

        # relation projection
        rel_emb_real_att=torch.tanh(self.linear_real(rel_emb_real)).unsqueeze(0) # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit_real, _ = self.multihead_attn_real(rel_emb_real_att, e1_emb_att_real,e1_emb_att_real)  # (1, batch_num, emb_dim)

        # gating
        e1_emb_real = self.emb_num_lit_real(e1_emb_real, e1_num_lit_real.squeeze(0))


        ###### tail real part
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att_real = numerical_literals * emb_att_real  # (att_num, num_entities, emb_dim)


        rel_emb_real_all_att = rel_emb_real.transpose(0,1).repeat(1,e2_emb_att_real.shape[1], 1)

        # multi-head attention
        e2_multi_num_lit_real, _ = self.multihead_attn_real(rel_emb_real_all_att, e2_emb_att_real, e2_emb_att_real)

        # gating
        e2_multi_emb_real = e2_multi_emb_real.unsqueeze(0).repeat(e1_emb_real.shape[0],1,1)  # (batch_num, num_entities, emb_dim)
        e2_multi_emb_real = self.emb_num_lit_real2(e2_multi_emb_real, e2_multi_num_lit_real)


        ###### head imaginary part
        emb_att_img = self.emb_att_img.unsqueeze(1)
        e1_emb_att_img = e1_num_lit * emb_att_img

        # relation projection
        rel_emb_img_att = torch.tanh(self.linear_img(rel_emb_img))

        # multi-head attention
        e1_num_lit_img, _ = self.multihead_attn_img(rel_emb_img_att.unsqueeze(0), e1_emb_att_img, e1_emb_att_img)

        # gating
        e1_emb_img = self.emb_num_lit_img(e1_emb_img, e1_num_lit_img.squeeze(0))


        ###### tail imaginary part
        e2_emb_att_img = numerical_literals * emb_att_img  # (att_num, num_entities, emb_dim)

        rel_emb_img_all_att = rel_emb_img_att.transpose(0,1).repeat(1,e2_emb_att_img.shape[1], 1)

        e2_multi_num_lit_img, _ = self.multihead_attn_img(rel_emb_img_all_att, e2_emb_att_img, e2_emb_att_img)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb_img = e2_multi_emb_img.unsqueeze(0).repeat(e1_emb_img.shape[0] ,1, 1)  # (batch_num, num_entities, emb_dim)
        e2_multi_emb_img = self.emb_num_lit_img2(e2_multi_emb_img, e2_multi_num_lit_img)


        # End literals
        e1_emb_real = self.inp_drop(e1_emb_real)
        rel_emb_real = self.inp_drop(rel_emb_real.squeeze())
        e1_emb_img = self.inp_drop(e1_emb_img)
        rel_emb_img = self.inp_drop(rel_emb_img)


        realrealreal = torch.bmm((e1_emb_real * rel_emb_real).unsqueeze(1), e2_multi_emb_real.transpose(1,2)).squeeze()
        realimgimg = torch.bmm((e1_emb_real * rel_emb_img).unsqueeze(1), e2_multi_emb_img.transpose(1,2)).squeeze()
        imgrealimg = torch.bmm((e1_emb_img * rel_emb_real).unsqueeze(1), e2_multi_emb_img.transpose(1,2)).squeeze()
        imgimgreal = torch.bmm((e1_emb_img * rel_emb_img).unsqueeze(1), e2_multi_emb_real.transpose(1,2)).squeeze()

        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)

        return pred

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

class RAKGEModel(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(RAKGEModel, self).__init__()

        self.bceloss = torch.nn.BCELoss()


        self.p = params
        self.init_embed = get_param((num_ents, self.p.init_dim))
        # self.ent_emb = get_param((num_ents, 190))
        # self.index_emb = get_param((num_ents, 10*2))

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num
        self.num_ents = num_ents

        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)
        self.multihead_attn2 = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.emb_num_lit2 = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.linear = nn.Linear(self.emb_dim, self.att_dim)

        #self.prj_path = Path(__file__).parent.resolve()
        # save_root = Path(__file__).parent.resolve() / 'weight.pt'
        #
        #
        # init_embed = torch.load(save_root)
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)



        # self.init_embed = init_embed
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))
        #numerical_literals = np.where(numerical_literals > 0, numerical_literals + 1, 0.01)
        #self.numerical_literals = F.dropout(torch.from_numpy(numerical_literals).cuda(), p=0.5)
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        self.init_rel = get_param((num_rels * 2, self.p.init_dim))

        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.inp_drop = nn.Dropout(p=self.p.input_drop)
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_tail = nn.CosineSimilarity(dim=2, eps=1e-6)



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
            'd': nn.Dropout(self.p.input_drop),
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


    def our_loss(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):

        prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
        # print(1, prob.shape)
        e2_multi_prob = e2_multi * prob
        # print(e2_multi.shape)
        # print(e2_multi_prob.shape)
        C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
        positive_mean = C.sum(dim=1) / e2_multi_prob.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)
        #print(positive_mean)

        alpha, beta = np.random.uniform(0, 1.0), np.random.uniform(0, 1.0)
        positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb
        #print(e2_multi.shape)



        # # # negative meadn
        e2_multi_inv = 1 - e2_multi
        e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)

        N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
        negative_mean = N.sum(dim=1) / e2_multi_inv.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)

        negative_mean = beta * negative_mean + (1 - beta) * e1_emb

        pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1,keepdim=True)
        neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)
        # print(pos_score2.shape)


        loss = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))

        e2_multi = e2_multi * prob

        head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1) * e2_multi
        # print('e1',e1_emb.shape)
        tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        score = F.logsigmoid(tail_score - head_score) * e2_multi

        # tail_score = (tail_score.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1))
        # print('h', head_score.shape)
        # print(tail_score.shape)
        loss = loss + (-1.0) * torch.mean((score.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1)))




        return loss

    def our_loss3(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):

        prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
        # print(1, prob.shape)
        e2_multi_prob = e2_multi * prob
        # print(e2_multi.shape)
        # print(e2_multi_prob.shape)
        P = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
        # print(P.shape)
        # positive_mean = C.sum(dim=1) / e2_multi_prob.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)
        #print(positive_mean)

        alpha = torch.rand(P.shape[0], P.shape[1],1).cuda()
        beta = torch.rand(P.shape[0], P.shape[1], 1).cuda()
        positive_mean = alpha * P + (1 - alpha) * e1_emb.unsqueeze(1)
        #print(e2_multi.shape)



        # # # negative meadn
        e2_multi_inv = 1 - e2_multi
        e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)

        N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
        # negative_mean = N.sum(dim=1) / e2_multi_inv.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)

        negative_mean = beta * N + (1 - beta) * e1_emb.unsqueeze(1)

        pos_score2 = torch.norm((e1_emb.unsqueeze(1) + rel_emb.unsqueeze(1) - positive_mean), p=1, dim=2)
        neg_score2 = torch.norm((e1_emb.unsqueeze(1) + rel_emb.unsqueeze(1) - negative_mean), p=1, dim=2)
        # print(pos_score2.shape)

        pos = (pos_score2 * e2_multi).sum(-1).unsqueeze(-1)/(e2_multi.sum(-1).unsqueeze(-1))

        # print(neg_score2.shape)
        # print(e2_multi.shape)
        neg = (neg_score2 * (1-e2_multi)).sum(-1).unsqueeze(-1)/(e2_multi_inv_partial.sum(-1).unsqueeze(-1))


        loss = (-1.0) * torch.mean(F.logsigmoid(neg - pos))

        # prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()


        # print(loss.shape)

        return loss

    def our_loss2(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):
        prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()

        e2_multi = e2_multi * prob

        head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1)*e2_multi
        # print('e1',e1_emb.shape)
        tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        score = F.logsigmoid(tail_score - head_score)*e2_multi

        # tail_score = (tail_score.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1))
        # print('h', head_score.shape)
        # print(tail_score.shape)
        loss = (-1.0) * torch.mean((score.sum(-1).unsqueeze(-1))/(e2_multi.sum(-1).unsqueeze(-1)))
        # print(loss.shape)
        return loss

    def our_loss(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):
        C = e2_multi_emb * e2_multi.unsqueeze(-1)
        positive_mean = C.sum(dim=1) / e2_multi.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)
        alpha = np.random.uniform(0, 0.1)
        positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb


        # # # negative meadn
        e2_multi_inv = 1 - e2_multi
        e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)

        N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
        negative_mean = N.sum(dim=1) / (e2_multi_inv.sum(-1).unsqueeze(-1)+ 10e-8)  # (batch_num, emb_dim)

        beta = np.random.uniform(0, 0.1)
        negative_mean = beta * negative_mean + (1 - beta) * e1_emb

        pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1,keepdim=True)
        neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)


        loss = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))

        return loss

    def our_loss(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):
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


    def our_loss1(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):

        prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()

        e2_multi_prob = e2_multi * prob

        C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
        positive_mean = C.sum(dim=1) / e2_multi_prob.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)
        #print(positive_mean)

        # alpha, beta = np.random.uniform(0, 1.0), np.random.uniform(0, 1.0)
        # positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb
        #print(e2_multi.shape)



        # # # negative meadn
        e2_multi_inv = 1 - e2_multi
        e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)

        N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
        negative_mean = N.sum(dim=1) / e2_multi_inv.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)

        # negative_mean = beta * negative_mean + (1 - beta) * e1_emb

        pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1,keepdim=True)
        neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)


        loss = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))

        return loss



class TransE_Gate_att(RAKGEModel):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, numerical_literals, params)
        self.loop_emb = get_param([1, self.p.init_dim])

    def forward(self, g, sub, rel, obj, e2_multi, pos_neg):
        # x_h = torch.cat([self.index_emb[:,:10], self.ent_emb], self.index_emb.ndimension() - 1)
        # x_t = torch.cat([self.index_emb[:,10:], self.ent_emb], self.index_emb.ndimension() - 1)

        x_h = self.init_embed-self.loop_emb
        x_t = self.init_embed-self.loop_emb
        r = self.init_rel

        e1_emb = torch.index_select(x_h, 0, sub) #(batch, init_dim)
        rel_emb = torch.index_select(r, 0, rel) #(batch, init_dim)
        e2_emb = torch.index_select(x_t, 0, obj)

        e2_multi_emb = x_t #(entity_num, init_dim)

        # e1_emb[:, :10] = 0
        # e2_multi_emb[:, 10:20] = 0
        numerical_literals = self.numerical_literals

        # Begin literals
        e1_num_lit = torch.index_select(numerical_literals, 0, sub)  # (batch_size, att_dim)

        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb.squeeze(0))).unsqueeze(0)  # (1, batch_num, att_dim)
        # rel_emb_att= rel_emb.unsqueeze(0)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # e1_emb = e1_num_lit.squeeze(0)

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

        e1_emb = e1_emb / torch.norm(e1_emb, p=2, dim=1, keepdim=True)
        # rel_emb = rel_emb/torch.norm(rel_emb, p=2, dim=1,keepdim=True)
        e2_multi_emb = e2_multi_emb / torch.norm(e2_multi_emb, p=2, dim=2, keepdim=True)

        # e2_multi_emb = e2_multi_num_lit
        #print(e2_multi_emb.shape)
        e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb,  self.x_ops, self.r_ops)


        if self.p.scale > 0 and self.p.dataset == 'Comp7':
            loss = self.our_loss(e1_emb,  rel_emb,e2_emb, e2_multi, e2_multi_emb)
            # print(loss)
            # loss = self.our_loss(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb)

        elif self.p.scale > 0 :
            loss = self.our_loss1(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb)
        else:
            loss = 0

        a = 0.3







        obj_emb = e1_emb + rel_emb # (batch, dim)


        # x = self.p.gamma - \
        #     torch.norm(e1_emb.unsqueeze(1)+ (1-a)*rel_emb.unsqueeze(1) - e2_multi_emb + a * self.linear1(e1_num_lit.squeeze(0).unsqueeze(1)-e2_multi_num_lit), p=1, dim=2)
        x = self.p.gamma - \
            torch.norm(obj_emb.unsqueeze(1)  - e2_multi_emb , p=1, dim=2)

        score = torch.sigmoid(x)
        # print(score.shape)

        return score, loss


class DistMult_Gate_att(RAKGEModel):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, numerical_literals, params)
        self.loop_emb = get_param([1, self.p.init_dim])

    def forward(self, g, sub, rel, obj, e2_multi, pos_neg):
        x_h = self.init_embed-self.loop_emb
        x_t = self.init_embed-self.loop_emb
        r = self.init_rel

        e1_emb = torch.index_select(x_h, 0, sub) #(batch, init_dim)
        rel_emb = torch.index_select(r, 0, rel) #(batch, init_dim)
        e2_emb = torch.index_select(x_t, 0, obj)

        e2_multi_emb = x_t #(entity_num, init_dim)
        numerical_literals = self.numerical_literals
        ###### head
        # Begin literals
        e1_num_lit = torch.index_select(numerical_literals, 0, sub)   # (batch_size, att_dim)


        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb.squeeze(0))).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze(0))  # (batch_num, emb_dim)


        ###### tail
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0,1).repeat(1,e2_emb_att.shape[1],1)
        e2_multi_num_lit, _ = self.multihead_attn(rel_emb_all_att, e2_emb_att, e2_emb_att) # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb, self.x_ops, self.r_ops)



        loss = self.our_loss(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_num_lit)


        e1 = e1_emb * rel_emb  # (batch_num, emb_dim)
        e1 = e1.unsqueeze(1)  # (batch_num, 1, emb_dim)
        pred = torch.matmul(e1, e2_multi_emb.transpose(1, 2))  # (batch_num, 1, num_entities) DistMult Score Function 적용
        pred = pred.squeeze()  # (batch_num, num_entities)
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)


        return pred, loss

'''class ConvE_Gate_att(RAKGEModel):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, numerical_literals,params)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.conve_hid_drop)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, g, sub, rel, obj, e2_multi, pos_neg):
        x_h = self.init_embed
        x_t = self.init_embed
        x = self.init_embed
        r = self.init_rel


        #x_h, x_t, r = self.exop(x, r, self.x_ops, self.r_ops)
        #e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb, self.x_ops, self.r_ops)

        e1_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        e2_emb = torch.index_select(x_t, 0, obj)
        # ---------------------------------
        e2_multi_emb = x_t

        e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb, self.x_ops, self.r_ops)


        ###### head
        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, sub)  # (batch_size, att_num)

        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb.squeeze(0))).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze(0))  # (batch_num, emb_dim)



        ###### tail
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)
        e2_multi_num_lit, _ = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                  e2_emb_att)  # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)





        loss = self.our_loss(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb)

        stk_inp = self.concat(e1_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.matmul(x.unsqueeze(1), e2_multi_emb.transpose(1, 2)).squeeze(1)


        #x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score, loss

class ConvE_Gate_att(torch.nn.Module):

    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(ConvE_Gate_att, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.emb_dim))
        self.emb_rel = get_param((num_rels * 2, self.emb_dim))


        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()



        # relation projection
        self.linear = nn.Linear(self.emb_dim, self.att_dim)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)
        # self.multihead_attn2 = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.emb_num_lit2 = Gate(self.att_dim + self.emb_dim, self.emb_dim)

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



    def forward(self, g, e1, rel, obj, e2_multi, pos_neg):
        e1_emb = torch.index_select(self.emb_e, 0, e1).view(-1, self.emb_dim)
        rel_emb = torch.index_select(self.emb_rel, 0, rel).view(-1, self.emb_dim)
        # ---------------------------------
        e2_multi_emb = self.emb_e

        ###### head
        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)  # (batch_size, att_num)

        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb.squeeze(0))).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze(0))  # (batch_num, emb_dim)

        ###### tail
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)
        e2_multi_num_lit, _ = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                  e2_emb_att)  # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

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
        x = torch.matmul(x.unsqueeze(1), e2_multi_emb.transpose(1, 2)).squeeze(1)

        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)

        return pred, 0

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)'''


class ConvE_Gate_att(torch.nn.Module):

    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(ConvE_Gate_att, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.emb_dim))
        self.emb_rel = get_param((num_rels * 2, self.emb_dim))


        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        # save_root = Path(__file__).parent.resolve() / 'weight.pt'

        # self.emb_e = torch.load(save_root)


        # relation projection
        self.linear = nn.Linear(self.emb_dim, self.att_dim)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)
        # self.multihead_attn2 = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.emb_num_lit2 = Gate(self.att_dim + self.emb_dim, self.emb_dim)

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



    def forward(self, g, e1, rel, obj, e2_multi, pos_neg):
        e1_emb = torch.index_select(self.emb_e, 0, e1).view(-1, self.emb_dim)
        rel_emb = torch.index_select(self.emb_rel, 0, rel).view(-1, self.emb_dim)
        # ---------------------------------
        e2_multi_emb = self.emb_e

        ###### head
        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)  # (batch_size, att_num)

        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)
        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb.squeeze(0))).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb_tmp = self.emb_num_lit(e1_emb, e1_num_lit.squeeze(0))  # (batch_num, emb_dim)

        ###### tail
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)
        e2_multi_num_lit, _ = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                  e2_emb_att)  # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        #e1_emb_tmp = self.inp_drop(e1_emb_tmp)
        e2_multi_emb = self.inp_drop(e2_multi_emb)

        # End literals
        e1_emb = e1_emb_tmp.view(len(e1), 1, 10, self.emb_dim//10)
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
        x_tmp = F.relu(x)
        x = torch.matmul(x_tmp.unsqueeze(1), e2_multi_emb.transpose(1, 2)).squeeze(1)

        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)


        if self.training:

            # C = e2_multi_emb * e2_multi.unsqueeze(-1)
            # positive_mean = C.sum(dim=1) / (e2_multi.sum(-1) + 1e-8).unsqueeze(-1)  # (batch_num, emb_dim)
            # alpha, beta = np.random.uniform(0, 0.5), np.random.uniform(0, 0.5)
            # positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb_tmp

            # # # negative meadn
            prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
            e2_multi_prob = e2_multi * prob
            C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
            positive_mean = C.sum(dim=1) / e2_multi_prob.sum(-1).unsqueeze(-1)  # (batch_num, emb_dim)

            alpha, beta = np.random.uniform(0, 1.0), np.random.uniform(0, 1.0)
            positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb_tmp



            e2_multi_inv = 1 - e2_multi
            e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)

            N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
            negative_mean = N.sum(dim=1) / (e2_multi_inv_partial.sum(-1) + 1e-8).unsqueeze(-1)  # (batch_num, emb_dim)
            #
            negative_mean = beta * negative_mean + (1 - beta) * e1_emb_tmp

            pos_score2 = (x_tmp*positive_mean).sum(1)
            neg_score2 = (x_tmp*negative_mean).sum(1)

            loss = (-1.0) * torch.mean(F.logsigmoid(pos_score2 - neg_score2))

            return pred,loss

        else:
            return pred, 0


    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class TransE_Gate_att(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(TransE_Gate_att, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels, self.p.init_dim))

        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_tail = nn.CosineSimilarity(dim=2, eps=1e-6)

        # print(self.numerical_literals)

        # relation projection
        self.linear = nn.Linear(self.emb_dim, self.att_dim)
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.emb_num_lit2 = Gate(self.att_dim + self.emb_dim, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

    def forward(self, g, e1, rel,e2, e2_multi, pos_neg=None):
        alpha = torch.index_select(self.alpha,0,rel)
        e1_emb = torch.index_select(self.emb_e, 0, e1)  # (batch_size, emb_dim)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)  # (batch_size, emb_dim)
        e2_multi_emb = self.emb_e

        ###### head
        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)  # (batch_size, att_dim)
        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)

        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb)).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze())  # (batch_num, emb_dim)

        ###### tail
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)
        e2_multi_num_lit, attention_matrix = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                                 e2_emb_att)  # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit2(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        ########## normalize 효과 별로 안 좋음 ##########
        # e1_emb = e1_emb / torch.norm(e1_emb, p=2, dim=1, keepdim=True)
        # rel_emb = rel_emb / torch.norm(rel_emb, p=2, dim=1, keepdim=True)
        # e2_multi_emb = e2_multi_emb / torch.norm(e2_multi_emb, p=2, dim=2, keepdim=True)

        e1_emb = self.inp_drop(e1_emb)
        # rel_emb = self.inp_drop(rel_emb)
        e2_multi_emb = self.inp_drop(e2_multi_emb)

        # score function
        score = self.p.gamma - \
                torch.norm(((e1_emb + rel_emb).unsqueeze(1) - e2_multi_emb), p=1, dim=2)

        #내가 추가
        a = 0.3
        score = self.p.gamma - \
            torch.norm(e1_emb.unsqueeze(1) +  (1-a) * rel_emb.unsqueeze(1) - e2_multi_emb + a * self.linear1(
                e1_num_lit.squeeze(0).unsqueeze(1) - e2_multi_num_lit), p=1, dim=2)

        #

        # 유닛벡터로 추가
        # rel_emb1 = rel_emb / torch.norm(rel_emb, p=2, dim=1, keepdim=True)
        # print(rel_emb1.shape)
        # att = torch.norm(self.linear1(e1_num_lit.squeeze(0).unsqueeze(1) - e2_multi_num_lit), p=1, dim=2, keepdim=True)
        # print(att.shape)
        # score = self.p.gamma - \
        #         torch.norm(e1_emb.unsqueeze(1) + rel_emb1.unsqueeze(1)* att - e2_multi_emb, p=1, dim=2)
        # print((rel_emb1.unsqueeze(1)* att).shape)
        # print(score.shape)




        pred = torch.sigmoid(score)






        # loss2
        rand = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()

        pred2 = pred.clone()
        pred2 = pred2 / (torch.sum(pred2 * e2_multi, dim=1, keepdim=True) +  1e-8)

        # 방법 1: 역수 이용
        e2_multi_prob = e2_multi * (1.0 / (pred2 + 1e-8)) * rand

        # 방법 2: gaussian distribution 이용
        # e2_multi_prob = e2_multi*torch.exp((-1)*torch.square(1-pred2))*rand

        # 방법 3: mean과의 거리 이용
        # C = e2_multi_emb*e2_multi.unsqueeze(-1)
        # positive_mean_real = C.sum(dim=1)/e2_multi.sum(-1).unsqueeze(-1)
        # cosine_sim = torch.matmul(e2_multi_emb,positive_mean_real.unsqueeze(-1))
        # cosine_sim = cosine_sim.squeeze()
        # e2_multi_prob = e2_multi*torch.exp((-1)*cosine_sim)*rand

        # loss2 = 0



        C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
        positive_mean = C.sum(dim=1) / (e2_multi_prob.sum(-1).unsqueeze(-1)+1e-8)  # (batch_num, emb_dim)

        # head mixing
        alpha = torch.rand(e2_multi.shape[0], 1).cuda()
        positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb

        # negative mean
        e2_multi_inv = 1 - e2_multi
        e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)
        N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
        negative_mean = N.sum(dim=1) / (e2_multi_inv_partial.sum(-1).unsqueeze(-1)+1e-8)  # (batch_num, emb_dim)

        # mixing
        beta = torch.rand(e2_multi.shape[0], 1).cuda()
        negative_mean = beta * negative_mean + (1 - beta) * e1_emb


        # loss2
        pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1)
        neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)
        loss2 = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2 ))

        # loss2=0
        #내가 추가
        # head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1) * e2_multi
        # tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        # score = F.logsigmoid(head_score - tail_score) * e2_multi
        #
        # loss2 = loss2 + 0.1 * (-1.0) * torch.mean((score.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1)))

        # 방법 4: entropy loss 추가
        # pred2 = pred.clone()
        # log_pred = torch.where(e2_multi==1, torch.log(pred2),0)
        # loss2 = torch.mean(pred2*log_pred*e2_multi)

        # 방법 5: KL-divergence loss
        # pred2 = pred.clone()
        # target = torch.ones(e2_multi.shape[0],e2_multi.shape[1]).cuda()
        # target = target/torch.sum(e2_multi, dim=1, keepdim=True)
        # loss2 = torch.mean(torch.sum(e2_multi*target*torch.log(target/(pred2+1e-8)), dim=1))

        autograd.set_detect_anomaly(True)
        return pred, loss2

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

    def our_loss(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb):
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

        # pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1,keepdim=True)
        # neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)
        #
        #
        # loss = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))

        e2_multi = e2_multi * prob
        loss = 0

        head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1) * e2_multi
        # print('e1',e1_emb.shape)
        tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        score = F.logsigmoid(tail_score - head_score) * e2_multi

        loss = loss + (-1.0) * torch.mean((score.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1)))




        return loss


class TransE_Gate_att(nn.Module):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(TransE_Gate_att, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim
        self.att_dim = self.p.att_dim
        self.head_num = self.p.head_num

        # entity, relation embedding table
        self.emb_e = get_param((num_ents, self.p.init_dim))
        self.emb_rel = get_param((num_rels, self.p.init_dim))

        # attribute embedding table
        self.num_att = numerical_literals.shape[1]
        self.emb_att = get_param((self.num_att, self.att_dim))

        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.cos_tail = nn.CosineSimilarity(dim=2, eps=1e-6)

        # print(self.numerical_literals)

        # relation projection
        self.linear = nn.Linear(self.emb_dim, self.att_dim)
        self.linear1 = nn.Linear(self.emb_dim, self.emb_dim)

        # MultiheadAttention
        self.multihead_attn = nn.MultiheadAttention(self.att_dim, self.head_num, batch_first=False)

        # gating
        self.emb_num_lit = Gate(self.att_dim + self.emb_dim, self.emb_dim)
        self.emb_num_lit2 = Gate(self.att_dim + self.emb_dim, self.emb_dim)

        self.bias = nn.Parameter(torch.zeros(num_ents))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

    def forward(self, g, e1, rel, e2, e2_multi, pos_neg=None):
        e1_emb = torch.index_select(self.emb_e, 0, e1)  # (batch_size, emb_dim)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)  # (batch_size, emb_dim)
        e2_multi_emb = self.emb_e

        ###### head
        # Begin literals
        e1_num_lit = torch.index_select(self.numerical_literals, 0, e1)  # (batch_size, att_dim)
        e1_num_lit = e1_num_lit.transpose(0, 1).unsqueeze(2)  # (att_num, batch_num, 1)

        emb_att = self.emb_att.unsqueeze(1)  # (att_num, 1, att_dim)
        e1_emb_att = e1_num_lit * emb_att  # (att_num, batch_num, att_dim)

        # relation projection
        rel_emb_att = torch.tanh(self.linear(rel_emb)).unsqueeze(0)  # (1, batch_num, att_dim)

        # multi-head attention
        e1_num_lit, _ = self.multihead_attn(rel_emb_att, e1_emb_att, e1_emb_att)  # (1, batch_num, att_dim)

        # gating
        e1_emb = self.emb_num_lit(e1_emb, e1_num_lit.squeeze())  # (batch_num, emb_dim)

        ###### tail
        # begin literals
        numerical_literals = self.numerical_literals.transpose(0, 1).unsqueeze(2)  # (att_num, num_entities, 1)
        e2_emb_att = numerical_literals * emb_att  # (att_num, num_entities, emb_dim)

        # multi-head attention
        rel_emb_all_att = rel_emb_att.transpose(0, 1).repeat(1, e2_emb_att.shape[1], 1)
        e2_multi_num_lit, attention_matrix = self.multihead_attn(rel_emb_all_att, e2_emb_att,
                                                                 e2_emb_att)  # (batch_num, num_entities, emb_dim)

        # gating
        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, e2_emb_att.shape[1], self.emb_dim)  # (batch_num, num_entities, emb_dim)

        e2_multi_emb = self.emb_num_lit2(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        e1_emb = self.inp_drop(e1_emb)
        # rel_emb = self.inp_drop(rel_emb)
        e2_multi_emb = self.inp_drop(e2_multi_emb)

        # score function
        score = self.p.gamma - \
                torch.norm(((e1_emb + rel_emb).unsqueeze(1) - e2_multi_emb), p=1, dim=2)

        pred = torch.sigmoid(score)

        # loss2
        rand = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()

        pred2 = pred.clone()
        pred2 = pred2 / (torch.sum(pred2 * e2_multi, dim=1, keepdim=True) + 1e-8)

        #
        e2_multi_prob = e2_multi * (1.0 / (pred2 + 1e-8)) * rand

        C = e2_multi_emb * e2_multi_prob.unsqueeze(-1)
        positive_mean = C.sum(dim=1) / (e2_multi_prob.sum(-1).unsqueeze(-1) + 1e-8)  # (batch_num, emb_dim)

        # head mixing
        alpha = torch.rand(e2_multi.shape[0], 1).cuda()
        positive_mean = alpha * positive_mean + (1 - alpha) * e1_emb

        # negative mean
        e2_multi_inv = 1 - e2_multi
        e2_multi_inv_partial = F.dropout(e2_multi_inv, p=0.5)
        N = e2_multi_emb * e2_multi_inv_partial.unsqueeze(-1)
        negative_mean = N.sum(dim=1) / (e2_multi_inv_partial.sum(-1).unsqueeze(-1) + 1e-8)  # (batch_num, emb_dim)

        # mixing
        beta = torch.rand(e2_multi.shape[0], 1).cuda()
        negative_mean = beta * negative_mean + (1 - beta) * e1_emb

        # loss2
        pos_score2 = torch.norm((e1_emb + rel_emb - positive_mean), p=1, dim=1)
        neg_score2 = torch.norm((e1_emb + rel_emb - negative_mean), p=1, dim=1)
        loss2 = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))


        autograd.set_detect_anomaly(True)
        return pred, loss2

    def calc_loss(self, pred, label):
        return self.loss(pred, label)

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)

