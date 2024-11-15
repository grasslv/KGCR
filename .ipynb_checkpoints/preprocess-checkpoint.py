"""
create on Aug 5, 2024

@author: ys
"""

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from collections import defaultdict
import os
from scipy.sparse import csr_matrix

random.seed(1234)

workdir = 'datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Ciao', help='dataset name: Ciao/Epinions')
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
args = parser.parse_args()

# load data
if args.dataset == 'Ciao':
	click_f = loadmat(workdir + 'Ciao/rating.mat')['rating']
	trust_f = loadmat(workdir + 'Ciao/trustnetwork.mat')['trustnetwork']
elif args.dataset == 'Epinions':
	click_f = np.loadtxt(workdir+'Epinions/ratings_data.txt', dtype = np.int32)
	trust_f = np.loadtxt(workdir+'Epinions/trust_data.txt', dtype = np.int32)
else:
	pass 

click_list = []
trust_list = []

u_items_list = []
u_users_list = []
u_users_items_list = []
i_users_list = []

user_count = 0
item_count = 0
rate_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	if args.dataset == 'Ciao':
		label = s[3]
	elif args.dataset == 'Epinions':
		label = s[2]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if label > rate_count:
		rate_count = label
	click_list.append([uid, iid, label])
print(f"user_count:{user_count}")
print(f"item_count:{item_count}")
pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

# remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
pos_list = list(set(pos_list))

# train, valid and test data split
random.shuffle(pos_list)
num_test = int(len(pos_list) * args.test_prop)
test_set = pos_list[:num_test]
valid_set = pos_list[num_test:2 * num_test]
train_set = pos_list[2 * num_test:]
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))


train_list = [row[:2] for row in train_set]
train_array = np.array(train_list)
np.save(workdir + args.dataset + '/train.npy',train_array)

test_list = [row[:2] for row in test_set]
test_array = np.array(test_list)
np.save(workdir + args.dataset + '/test.npy',test_array)

sorted_train_set = sorted(train_set, key=lambda x: x[0])
sorted_test_set = sorted(test_set, key=lambda x: x[0])

'''
u-i
'''
u_item_dict = defaultdict(list)
for i in sorted_train_set:
	if i[1] not in u_item_dict[i[0]]:
		u_item_dict[i[0]].append(i[1])
for i in sorted_test_set:
	if i[1] not in u_item_dict[i[0]]:
	    u_item_dict[i[0]].append(i[1])
np.save(workdir + args.dataset + '/user_item_dict.npy', u_item_dict)

u_item_dict_train = defaultdict(list)
for i in sorted_train_set:
	if i[1] not in u_item_dict_train[i[0]]:
		u_item_dict_train[i[0]].append(i[1])
np.save(workdir + args.dataset + '/user_item_dict_train.npy', u_item_dict_train)

'''
i-u
'''
i_user_dict = defaultdict(list)
for i in sorted_train_set:
	if i[0] not in i_user_dict[i[1]]:
		i_user_dict[i[1]].append(i[0])
for i in sorted_test_set:
	if i[0] not in i_user_dict[i[1]]:
	    i_user_dict[i[1]].append(i[0])
np.save(workdir + args.dataset + '/i_user_dict.npy', i_user_dict)

i_user_dict_train = defaultdict(list)
for i in sorted_train_set:
	if i[0] not in i_user_dict_train[i[1]]:
		i_user_dict_train[i[1]].append(i[0])
#np.save(workdir + args.diataset + '/i_user_dict_train.npy', i_user_dict_train)


'''
u-u
'''
for s in trust_f:
	uid = s[0]
	fid = s[1]
	if uid > user_count or fid > user_count:
		continue
	trust_list.append([uid, fid])
sorted_trust_list = sorted(trust_list, key=lambda x: x[0])

u_user_dict =defaultdict(list)
for i in sorted_trust_list:
	if i[1] not in u_user_dict[i[0]]:
		u_user_dict[i[0]].append(i[1])
np.save(workdir + args.dataset + '/u_e_dict.npy', u_user_dict)
for u1,us in u_user_dict.items():
	for u in us:
		u_users_list.append([u1,u])
u_users_array = np.array(u_users_list)
u_users_array = u_users_array.astype(np.int32)
np.save(workdir + args.dataset + '/u_u_list.npy',u_users_array)



'''
i-i
'''
def build_sparse_matrix(dict_data, num_rows, num_cols):
	row_indices = []
	col_indices = []
	data = []
	for row, cols in dict_data.items():
		for col in cols:
			row_indices.append(row)
			col_indices.append(col)
			data.append(1)  # 稀疏矩阵的元素为1
	matrix = csr_matrix((data, (row_indices, col_indices)), shape=(num_rows, num_cols))
	return matrix

def process_data(i_user_dict_train, u_user_dict, u_item_dict_train):
    # 计算矩阵大小
    num_items = item_count+1
    num_users = user_count+1
    
    # 构建 i-u 和 u-u 的稀疏矩阵
    i_user_matrix = build_sparse_matrix(i_user_dict_train, num_items, num_users)
    u_user_matrix = build_sparse_matrix(u_user_dict, num_users, num_users)
    u_item_matrix = build_sparse_matrix(u_item_dict_train, num_users, num_items)
    # 计算 i-u * u-u
    i_u_u_matrix = i_user_matrix.dot(u_user_matrix)
	# 计算i-u * u-i
    i_i_matrix = i_u_u_matrix.dot(u_item_matrix)

    # 将 i-u * u-i * i-i 结果转换为字典
    i_i_dict = defaultdict(set)
    for i in range(i_i_matrix.shape[0]):
        items = i_u_u_matrix[i].indices
        i_i_dict[i].update(items)
    
    return i_i_dict

# 执行处理
i_item_dict = process_data(i_user_dict_train, u_user_dict, u_item_dict_train)

# 转换为列表
i_item_list = [[i1, item] for i1, items in i_item_dict.items() for item in items]
i_item_array = np.array(i_item_list)

# 保存结果
np.save(workdir + args.dataset + '/i_item_list_1.npy', i_item_array)






#----------------memmap-------------------#
# # 使用集合初始化 defaultdict
# i_item_dict = defaultdict(set)
# i_u_u_dict = defaultdict(set)

# # 设置保存目录和文件名
# output_dir = os.path.join(workdir, args.dataset)
# os.makedirs(output_dir, exist_ok=True)

# # 临时输出文件路径
# temp_output_file = os.path.join(output_dir, 'i_item_list_temp.npy')


# # 1. i-u * u-u
# for item, users in tqdm(i_user_dict_train.items()):
#     users_set = set(users)  # 转换为集合以提高查找效率
#     for u1, u2 in u_user_dict.items():
#         if u1 in users_set:
#             i_u_u_dict[item].update(u2)  # 直接更新集合，无需去重
# # 2. i-u * u-i
# for item, users in tqdm(i_u_u_dict.items()):
#     users_set = set(users)  # 转换为集合以提高查找效率
#     for u, items in i_user_dict_train.items():
#         if u in users_set:
#             i_item_dict[item].update(items)

# # 获取所有 i-item 对的总数量
# total_items = sum(len(items) for items in i_item_dict.values())
# print(total_items)

# # 使用内存映射创建一个空的 .npy 文件
# final_output_file = os.path.join(output_dir, 'i_item_list.npy')
# i_item_array = np.memmap(final_output_file, dtype='int32', mode='w+', shape=(total_items, 2))

# # 填充内存映射数组
# index = 0
# for i1, items in tqdm(i_item_dict.items()):
#     for item in items:
#         i_item_array[index] = [i1, item]
#         index += 1

# # 确保所有更改已写入磁盘
# i_item_array.flush()

# print("结果已保存到", final_output_file)



# ------------------------全处理-----------------#
# 使用集合初始化 defaultdict
# i_item_dict = defaultdict(set)
# i_u_u_dict = defaultdict(set)

# # 1. i-u * u-u
# for item, users in tqdm(i_user_dict_train.items()):
#     users_set = set(users)  # 转换为集合以提高查找效率
#     for u1, u2 in u_user_dict.items():
#         if u1 in users_set:
#             u2_set = set(u2)  # 转换为集合以提高查找效率
#             i_u_u_dict[item].update(u2_set)  # 使用 update() 方法添加元素

# # 2. i-u * u-i
# for item, users in tqdm(i_u_u_dict.items()):
#     users_set = set(users)  # 转换为集合以提高查找效率
#     for u, items in u_item_dict_train.items():
#         if u in users_set:
#             i_item_dict[item].update(items)  # 使用 update() 方法添加元素

# # 转换为列表
# i_item_list = [[i1, item] for i1, items in i_item_dict.items() for item in items]
# i_item_array = np.array(i_item_list)

# # 保存结果
# np.save(workdir + args.dataset + '/i_item_list.npy', i_item_array)


# i_item_dict = defaultdict(list)
# i_u_u_dict = defaultdict(list)
# #1.i-u * u-u
# for item,users in tqdm(i_user_dict_train.items()):
# 	for u1,u2 in u_user_dict.items():
# 		if u1 in users:
# 			for u in u2:
# 				if u not in i_u_u_dict[item]:
# 					i_u_u_dict[item].append(u)
# #2.i-u * u-i
# for item,users in tqdm(i_u_u_dict.items()):
# 	for u, items in u_item_dict_train.items():
# 		if u in users:
# 			for i in items:
# 				if i not in i_item_dict[item]:
# 					i_item_dict[item].append(i)
# #np.save(workdir + args.dataset + '/i_i_dict.npy', i_item_dict)
# i_item_list = []
# for i1, items in i_item_dict.items():
# 	for item in items:
# 		i_item_list.append([i1,item])
# i_item_array = np.array(i_item_list)
# np.save(workdir + args.dataset + '/i_item_list.npy',i_item_array)
	



