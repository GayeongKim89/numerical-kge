import torch
from itertools import chain

from torch import nn

def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

class MTKGNN(torch.nn.Module):

    def __init__(self, num_entities, num_relations, numerical_literals, params=None):
        super(MTKGNN, self).__init__()

        self.bceloss = torch.nn.BCELoss()
        self.p = params

        self.device = "cuda"
        self.emb_dim = self.p.init_dim

        self.num_entities = num_entities
        self.num_relations = num_relations

        self.emb_e = get_param((num_entities, self.emb_dim))
        self.emb_rel = get_param((2*num_relations, self.emb_dim))

        # Literal
        # num_ent x n_num_lit
        self.numerical_literals = torch.from_numpy(numerical_literals).cuda()
        self.n_num_lit = self.numerical_literals.size(1)

        self.emb_attr = get_param((self.n_num_lit, self.emb_dim))


        self.rel_net = torch.nn.Sequential(
            torch.nn.Linear(3*self.emb_dim, 100, bias=False),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

        self.attr_net_left = torch.nn.Sequential(
            torch.nn.Linear(2*self.emb_dim, 100, bias=False),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

        self.attr_net_right = torch.nn.Sequential(
            torch.nn.Linear(2*self.emb_dim, 100, bias=False),
            torch.nn.Tanh(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(100, 1),
            torch.nn.Sigmoid()
        )

        self.rel_params = chain(self.emb_e, self.emb_rel)
        self.attr_params = chain(self.emb_e, self.emb_attr,
        self.attr_net_left, self.attr_net_right)

        # Dropout + loss
        self.bias = nn.Parameter(torch.zeros(num_entities))
        self.inp_drop = nn.Dropout(p=self.p.input_drop)

        self.loss_rel = torch.nn.BCELoss()
        self.loss_attr = torch.nn.MSELoss()


    def forward(self, e1, rel):

        e1_emb = torch.index_select(self.emb_e, 0, e1)  # (batch_size, emb_dim)
        rel_emb = torch.index_select(self.emb_rel, 0, rel)  # (batch_size, emb_dim)

        e2_multi_emb = self.emb_e

        e1_emb_all = e1_emb.unsqueeze(1).repeat(1,self.num_entities,1)
        rel_emb_all = rel_emb.unsqueeze(1).repeat(1,self.num_entities,1)

        e2_emb_all = e2_multi_emb.repeat(e1_emb.shape[0], 1)  # (batch_num * num_entities, emb_dim)
        e2_emb_all = e2_emb_all.view(-1, self.num_entities, self.emb_dim)  # (batch_num, num_entities, emb_dim)

        inputs = torch.cat([e1_emb_all, e2_emb_all, rel_emb_all], dim=2) # (batch_num, num_entities, 3*emb_dim)

        pred = self.rel_net(inputs)   # (batch_num, num_entities, 1)
        pred = pred.squeeze()       # (batch_num, num_entities)

        #pred = torch.mm(e1_emb*rel_emb, self.emb_e.transpose(1,0))
        #pred = torch.sigmoid(pred)

        return pred

    def forward_attr(self, e, mode='left'):
        assert mode == 'left' or mode == 'right'


        # Sample one numerical literal for each entity
        #e_attr = torch.index_select(self.numerical_literals, 0, e)  # (batch_num, n_num_lit)

        if mode == 'left':
            e_emb = torch.index_select(self.emb_e, 0, e)  # (batch_num, emb_dim)
            m = e_emb.shape[0]
            idxs = torch.randint(self.n_num_lit, size=(m,)).cuda()
            attr_emb = torch.index_select(self.emb_attr, 0, idxs)   # (batch_num, emb_dim)
            inputs = torch.cat([attr_emb, e_emb], dim=1)
            pred = self.attr_net_left(inputs)

            target = self.numerical_literals[range(m), idxs]

        elif mode == 'right':
            idxs = torch.randint(self.n_num_lit, size=(self.num_entities,)).cuda()
            attr_emb = torch.index_select(self.emb_attr, 0, idxs)  # (batch_num, emb_dim)
            inputs = torch.cat([attr_emb, self.emb_e], dim=1)
            pred = self.attr_net_right(inputs)

            target = self.numerical_literals[range(self.num_entities),idxs]



        pred = pred.squeeze()
        #pred = torch.sigmoid(pred)


        return pred, target

    def forward_AST(self):
        m = self.p.batch_size

        idxs_attr = torch.randint(self.n_num_lit, size=(1,)).cuda()
        #idxs_attr = torch.randint(self.n_num_lit, size=(m,)).cuda()
        idxs_ent = torch.randint(self.num_entities, size=(m,)).cuda()

        attr_emb = torch.index_select(self.emb_attr, 0, idxs_attr)
        ent_emb = torch.index_select(self.emb_e, 0, idxs_ent)


        attr_emb = attr_emb.repeat(m,1)
        inputs = torch.cat([attr_emb, ent_emb], dim=1)

        pred_left = self.attr_net_left(inputs)
        pred_right = self.attr_net_right(inputs)

        target = self.numerical_literals[idxs_ent, idxs_attr]

        pred_left = pred_left.squeeze()
        pred_right = pred_right.squeeze()

        return pred_left, pred_right, target