# new_kg_list.npy: <h, r, t>
# train_list.npy: <u, i>
# test_dict.npy: <u, i>
# user_item_dict.npy: <u,{i}> train&test
# user_item_dict_train.npy: <u,{i}> train
# user_entity_dict.npy: <u, {e}> train
# u_e_list.npy: <u, e> train


import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# def data_load(dataset):
#     if dataset == 'yelp2018':
#         num_a = 90961
#         num_u = 45919
#         num_i = 45538
#         num_r = 42
#     elif dataset == 'lastfm':
#         num_a = 58266
#         num_u = 23566
#         num_i = 48123
#         num_r = 9
#     elif dataset == 'amazon-book':
#         num_a = 88572
#         num_u = 70679
#         num_i = 24915
#         num_r = 39

#     dir_str = './datasets/' + dataset
#     kg_data = np.load(dir_str+'/new_kg_list.npy')
#     train_data = np.load(dir_str+'/train_list.npy')
#     test_data = np.load(dir_str+'/test_dict.npy', allow_pickle=True)
#     user_item_dict = np.load(dir_str+'/user_item_dict.npy', allow_pickle=True).item()
#     user_item_dict_train = np.load(dir_str+'/user_item_dict_train.npy', allow_pickle=True).item()
#     h_r_dict = np.load(dir_str+'/h_r_dict.npy', allow_pickle=True).item()
#     # u_e_list = np.load(dir_str+'/u_e_list.npy')
#     u_e_list = np.load(dir_str+'/u_e_list_new.npy')

#     kg_list = np.column_stack((kg_data[:,0], kg_data[:,2]))
#     relation_list = kg_data[:,1]
#     u_e_index = np.load(dir_str+'/all_u_a_list.npy').tolist()
#     u_e_value = np.load(dir_str+'/all_value_list.npy').tolist()
#     att_weight = torch.sparse_coo_tensor(u_e_index, u_e_value, (num_u, num_a))
#     print(num_a, num_u, num_i)


#     return train_data, test_data, kg_list, relation_list, u_e_list, user_item_dict, user_item_dict_train, h_r_dict, num_u, num_i, num_r, num_a, att_weight


def data_load(dataset):
    if dataset == 'Ciao':
        num_a = 7376
        num_u = 7376
        num_i = 106798
    elif dataset == 'Epinions':
        num_a = 49290
        num_u = 49290
        num_i = 139739
    dir_str = './datasets/' + dataset
    train_data = np.load(dir_str + '/train.npy')
    test_data = np.load(dir_str + '/test.npy', allow_pickle=True)
    u_u_list = np.load(dir_str + '/u_u_list.npy')
    user_item_dict = np.load(dir_str + '/user_item_dict.npy', allow_pickle=True).item()
    user_item_dict_train = np.load(dir_str + '/user_item_dict_train.npy', allow_pickle=True).item()
    # i_i_data = np.memmap(dir_str+'/i_item_list.npy', dtype='int32', mode='r',shape=(45238965,2))
    i_i_data = np.load(dir_str + '/i_item_list.npy')
    u_u_dict = np.load(dir_str + '/u_e_dict.npy', allow_pickle=True).item()
    i_u_data = np.load(dir_str + '/i_user_dict.npy', allow_pickle=True).item()
    print(num_a, num_u, num_i)
    # train_data:u-i
    # kg_list: u-u
    # u_e_list:i-i
    return train_data, test_data, u_u_list, i_i_data, user_item_dict, user_item_dict_train, u_u_dict, num_u, num_i, num_a


class TrainDataset(Dataset):
    def __init__(self, train_data, user_item_dict, num_i, num_u):
        self.train_data = train_data
        self.user_item_dict = user_item_dict
        self.all_item = set(range(num_u, num_u + num_i))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        user, item = self.train_data[index]
        while True:
            neg_item = random.sample(self.all_item, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break

        return torch.LongTensor([user, user]), torch.LongTensor([item, neg_item])


if __name__ == '__main__':
    train_data, val_data, val_label, test_data, test_label, user_att_dict, item_att_dict, user_item_dict, all_item, att_num, user_num, item_num = data_load(
        'LON')
    # train_dataset = MyDataset(train_data, user_att_dict, item_att_dict, all_item, user_item_dict)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=10)
    # val_dataset = VTDataset(val_data, val_label, user_att_dict, item_att_dict)
    # val_dataloader = DataLoader(val_dataset, batch_size=51)
    # for a, u, i, l in val_dataloader:
    #     print(a.size(), u.size(), i.size(), l.size())
    #     print(l)
