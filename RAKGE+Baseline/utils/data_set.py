from torch.utils.data import Dataset
import numpy as np
import torch
import random

class TrainDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TrainDataset, self).__init__()
        self.p = params
        self.triplets = triplets
        self.label_smooth = params.lbl_smooth
        self.num_ent = num_ent
        self.neg = params.neg

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        pos_neg= self.get_pos_neg(triple, label)
        label = self.get_label(label)
        if self.label_smooth != 0.0:
            label = (1.0 - self.label_smooth) * label + (1.0 / self.num_ent)

        return triple, label, pos_neg

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

    def get_pos_neg(self, triple, label):


        if len(label)==1:
            a=label[0]
            a = np.array([a])

        else:

            label2= np.setdiff1d(label, triple[2])

            a=np.random.choice(label2,1,replace=False)
            a = np.array(a)

        all_entity = np.setdiff1d(np.arange(self.num_ent), label)
        b = np.random.choice(all_entity, self.neg, replace=False)


        return torch.tensor(np.concatenate([a, b]))





class TestDataset(Dataset):
    def __init__(self, triplets, num_ent, params):
        super(TestDataset, self).__init__()
        self.triplets = triplets
        self.num_ent = num_ent
        self.neg = params.neg

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, item):
        ele = self.triplets[item]
        triple, label = torch.tensor(ele['triple'], dtype=torch.long), np.int32(ele['label'])
        pos_neg = self.get_pos_neg(triple, label)
        label = self.get_label(label)
        return triple, label, pos_neg

    def get_label(self, label):
        """
        get label corresponding to a (sub, rel) pair
        :param label: a list containing indices of objects corresponding to a (sub, rel) pair
        :return: a tensor of shape [nun_ent]
        """
        y = np.zeros([self.num_ent], dtype=np.float32)
        y[label] = 1
        return torch.tensor(y, dtype=torch.float32)

    def get_pos_neg(self, triple, label):



        if len(label)==1:
            a=label[0]
            a = np.array([a])

        else:

            label2= np.setdiff1d(label, triple[2])

            a=np.random.choice(label2,1,replace=False)
            a = np.array(a)

        all_entity = np.setdiff1d(np.arange(self.num_ent), label)
        b = np.random.choice(all_entity, self.neg, replace=False)


        return torch.tensor(np.concatenate([a, b]))
