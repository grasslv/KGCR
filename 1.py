import torch
import numpy as np
import torch.nn as nn
data_path = "Ciao"
# 加载张量
item = torch.load('./datasets/'+data_path+'/item.pt')
user = torch.load('./datasets/'+data_path+'/user.pt')



user = user.weight
item = item.weight

id_emd = torch.cat((user,item),dim=0)


torch.save(id_emd,'./datasets/'+data_path+'/id.pt')