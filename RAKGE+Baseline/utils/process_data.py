# from collections import defaultdict as ddict
# import numpy as np
#
#
#
# def process(dataset, num_rel):
#     """
#     pre-process dataset
#     :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
#     :param num_rel: relation number
#     :return:
#     """
#     print(len(dataset['train']), len(dataset['valid']),len(dataset['test']))
#     threshold = 0.7
#     sr2o = ddict(set)
#     np.random.seed(100)
#     train_random = np.random.rand(len(dataset['train']), 2)
#     np.random.seed(101)
#     val_random = np.random.rand(len(dataset['valid']), 2)
#     np.random.seed(102)
#     test_random = np.random.rand(len(dataset['test']),2)
#
#     for i, (subj, rel, obj) in enumerate(dataset['train']):
#         a , b = train_random[i]
#         if a <= threshold:
#             sr2o[(subj, rel)].add(obj)
#         if b <= threshold:
#             sr2o[(obj, rel + num_rel)].add(subj)
#     sr2o_train = {k: list(v) for k, v in sr2o.items()}
#     '''for split in ['valid', 'test']:
#         for subj, rel, obj in dataset[split]:
#             sr2o[(subj, rel)].add(obj)
#             sr2o[(obj, rel + num_rel)].add(subj)'''
#     for j, (subj, rel, obj) in enumerate(dataset['valid']):
#         c, d = val_random[j]
#         if c <= threshold:
#             sr2o[(subj, rel)].add(obj)
#         if d <= threshold:
#             sr2o[(obj, rel + num_rel)].add(subj)
#     for k, (subj, rel, obj) in enumerate(dataset['test']):
#         e, f = test_random[k]
#         if e <= threshold:
#             sr2o[(subj, rel)].add(obj)
#         if f <= threshold:
#             sr2o[(obj, rel + num_rel)].add(subj)
#
#
#
#     sr2o_all = {k: list(v) for k, v in sr2o.items()}
#     triplets = ddict(list)
#
#     for (subj, rel), obj in sr2o_train.items():
#         triplets['train'].append({'triple': (subj, rel, obj[0]), 'label': sr2o_train[(subj, rel)]})
#         #원래 obj[0]대신 -1 사용
#     '''for split in ['valid', 'test']:
#         for subj, rel, obj in dataset[split]:
#             triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
#             triplets[f"{split}_head"].append(
#                 {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})'''
#     for j, (subj, rel, obj) in enumerate(dataset['valid']):
#         c, d = val_random[j]
#         if c<=threshold:
#             triplets["valid_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
#         if d<=threshold:
#             triplets["valid_head"].append(
#             {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
#     for k, (subj, rel, obj) in enumerate(dataset['test']):
#         e, f = test_random[k]
#         if e<=threshold:
#             triplets["test_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
#         if f<=threshold:
#             triplets["test_head"].append(
#             {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
#     triplets = dict(triplets)
#     return triplets

from collections import defaultdict as ddict


def process(dataset, num_rel):
    """
    pre-process dataset
    :param dataset: a dictionary containing 'train', 'valid' and 'test' data.
    :param num_rel: relation number
    :return:
    """
    sr2o = ddict(set)
    for subj, rel, obj in dataset['train']:
        sr2o[(subj, rel)].add(obj)
        # sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_train = {k: list(v) for k, v in sr2o.items()}
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            sr2o[(subj, rel)].add(obj)
            # sr2o[(obj, rel + num_rel)].add(subj)
    sr2o_all = {k: list(v) for k, v in sr2o.items()}
    triplets = ddict(list)

    for (subj, rel), obj in sr2o_train.items():
        triplets['train'].append({'triple': (subj, rel, obj[0]), 'label': sr2o_train[(subj, rel)]})
        #원래 obj[0]대신 -1 사용
    for split in ['valid', 'test']:
        for subj, rel, obj in dataset[split]:
            triplets[f"{split}_tail"].append({'triple': (subj, rel, obj), 'label': sr2o_all[(subj, rel)]})
            # triplets[f"{split}_head"].append(
            #     {'triple': (obj, rel + num_rel, subj), 'label': sr2o_all[(obj, rel + num_rel)]})
    triplets = dict(triplets)
    return triplets
