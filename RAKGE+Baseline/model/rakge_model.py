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
        # self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        # self.numerical_literals = nn.Parameter(torch.from_numpy(numerical_literals)).cuda()

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



    def our_loss(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb,pred2):
        rand = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()

        # pred2 = pred.clone()
        pred2 = pred2 / (torch.sum(pred2 * e2_multi, dim=1, keepdim=True) + 1e-8)

        # 방법 1: 역수 이용
        e2_multi_prob = e2_multi * (1.0 / (pred2 + 1e-8)) * rand

        # prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
        # e2_multi_prob = e2_multi * prob

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

        # 내가 추가
        # head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1) * e2_multi
        # tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        # score = F.logsigmoid(head_score - tail_score) * e2_multi
        # 수정 1
        # head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1)
        # tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        # score_pos = F.logsigmoid(head_score*e2_multi - tail_score) * e2_multi
        # score_neg = F.logsigmoid(tail_score - head_score * (1-e2_multi)) * (1-e2_multi)
        #
        #
        # loss = loss + 0 * (-1.0) * (torch.mean((score_pos.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1)+ 1e-8))
        #                               + torch.mean((score_neg.sum(-1).unsqueeze(-1)) / ((1-e2_multi).sum(-1).unsqueeze(-1)+ 1e-8)))


        return loss

    def our_loss_a(self, e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb,pred2,al, e1_num_lit, e2_num_lit):
        rand = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()

        # pred2 = pred.clone()
        pred2 = pred2 / (torch.sum(pred2 * e2_multi, dim=1, keepdim=True) + 1e-8)

        # 방법 1: 역수 이용
        e2_multi_prob = e2_multi * (1.0 / (pred2 + 1e-8)) * rand

        # prob = torch.rand(e2_multi.shape[0], e2_multi.shape[1]).cuda()
        # e2_multi_prob = e2_multi * prob

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
        # print(((e2_num_lit * e2_multi_prob.unsqueeze(-1) * alpha + (1-alpha) * e1_num_lit)*e1_num_lit).shape)
        # p_a = al * self.linear1(e1_emb @ positive_mean)
        # n_a = al * self.linear1(e1_emb @ negative_mean)

        c =self.linear1((e2_num_lit * e2_multi_prob.unsqueeze(-1) * alpha + (1-alpha) * e1_num_lit*e2_multi_prob.unsqueeze(-1))*e1_num_lit)
        pos_al = c.sum(dim=1) / e2_multi_prob.sum(-1).unsqueeze(-1)
        # print((al*pos_al).shape)
        # print(positive_mean.shape)
        # print(e1_emb.shape)
        # print(rel_emb.shape)

        n = self.linear1((e2_num_lit * e2_multi_inv_partial.unsqueeze(-1) * beta + (1 - beta) * e1_num_lit * e2_multi_inv_partial.unsqueeze(-1))*e1_num_lit)
        neg_al = n.sum(dim=1) / e2_multi_inv.sum(-1).unsqueeze(-1)



        pos_score2 = torch.norm((e1_emb + rel_emb + al*pos_al -  positive_mean), p=1, dim=1,keepdim=True)
        neg_score2 = torch.norm((e1_emb + rel_emb + al*neg_al - negative_mean), p=1, dim=1)


        loss = (-1.0) * torch.mean(F.logsigmoid(neg_score2 - pos_score2))

        # 내가 추가
        # head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1) * e2_multi
        # tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        # score = F.logsigmoid(head_score - tail_score) * e2_multi
        # 수정 1
        head_score = self.cos(e1_emb, rel_emb).unsqueeze(-1)
        tail_score = self.cos_tail(rel_emb.unsqueeze(1), e2_multi_emb)
        score_pos = F.logsigmoid(head_score*e2_multi - tail_score) * e2_multi
        score_neg = F.logsigmoid(tail_score - head_score * (1-e2_multi)) * (1-e2_multi)


        loss = loss + 0.3 * (-1.0) * (torch.mean((score_pos.sum(-1).unsqueeze(-1)) / (e2_multi.sum(-1).unsqueeze(-1)+ 1e-8)))
                                      #+ torch.mean((score_neg.sum(-1).unsqueeze(-1)) / ((1-e2_multi).sum(-1).unsqueeze(-1)+ 1e-8)))


        return loss




class TransE_Gate_att(RAKGEModel):
    def __init__(self, num_ents, num_rels, numerical_literals, params=None):
        super(self.__class__, self).__init__(num_ents, num_rels, numerical_literals, params)
        self.loop_emb = get_param([1, self.p.init_dim])



    def forward(self, g, sub, rel, obj, e2_multi, pos_neg):
        x_h = self.init_embed-self.loop_emb
        x_t = self.init_embed-self.loop_emb
        r = self.init_rel
        a = self.alpha
        s = self.s
        alpha = torch.index_select(a, 0, rel)
        e1_emb = torch.index_select(x_h, 0, sub) #(batch, init_dim)
        rel_emb = torch.index_select(r, 0, rel) #(batch, init_dim)
        e2_emb = torch.index_select(x_t, 0, obj)
        # scale = torch.index_select(s, 0, sub)
        # print('s',scale.shape)


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
        #변형1
        # a = 0.3
        #
        #
        # score = self.p.gamma - \
        #         torch.norm(e1_emb.unsqueeze(1) + (1 - a) * rel_emb.unsqueeze(1) - e2_multi_emb + a * self.linear1(
        #             e1_num_lit.squeeze(0).unsqueeze(1) - e2_multi_num_lit), p=1, dim=2)
        # score = torch.sigmoid(score)
        #변형2

        # al = alpha.unsqueeze(1) * self.linear1(e1_emb.unsqueeze(1) - e2_multi_emb)
        # al = alpha.unsqueeze(1) * self.linear1(e1_num_lit.squeeze(0).unsqueeze(1) - e2_multi_num_lit)
        # rel_s_emb = rel_emb.unsqueeze(1) + al
        # x = self.p.gamma - \
        #     torch.norm(e1_emb.unsqueeze(1) + rel_s_emb - e2_multi_emb, p=1, dim=2)
        # score = torch.sigmoid(x)

        # 변형3
        # al = rel_emb.unsqueeze(1) * self.linear1(e1_num_lit.squeeze(0).unsqueeze(1) - e2_multi_num_lit)
        # rel_s_emb = rel_emb.unsqueeze(1) + al
        # x = self.p.gamma - \
        #     torch.norm(e1_emb.unsqueeze(1) + rel_s_emb - e2_multi_emb, p=1, dim=2)
        # score = torch.sigmoid(x)



        loss = self.our_loss(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb, score)





        # loss = self.our_loss_a(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_emb, score,alpha, e1_num_lit.squeeze(0).unsqueeze(1), e2_multi_num_lit)

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

        e2_multi_emb = self.emb_num_lit2(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb, self.x_ops, self.r_ops)



        loss = self.our_loss(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_num_lit)


        e1 = e1_emb * rel_emb  # (batch_num, emb_dim)
        e1 = e1.unsqueeze(1)  # (batch_num, 1, emb_dim)
        pred = torch.matmul(e1, e2_multi_emb.transpose(1, 2))  # (batch_num, 1, num_entities) DistMult Score Function 적용
        pred = pred.squeeze()  # (batch_num, num_entities)
        pred += self.bias.expand_as(pred)
        pred = torch.sigmoid(pred)


        return pred, loss

class ConvE_Gate_att(RAKGEModel):
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


        e1_emb = torch.index_select(x_h, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)
        e2_emb = torch.index_select(x_t, 0, obj)
        # ---------------------------------
        e2_multi_emb = x_t


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

        e2_multi_emb = self.emb_num_lit2(e2_emb_all, e2_multi_num_lit)  # (batch_num, num_entities, emb_dim)

        loss = self.our_loss(e1_emb, rel_emb, e2_emb, e2_multi, e2_multi_num_lit)

        e1_emb, e2_multi_emb, rel_emb = self.exop_last(e1_emb, e2_multi_emb, rel_emb, self.x_ops, self.r_ops)

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