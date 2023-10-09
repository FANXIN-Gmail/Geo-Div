# -- coding:UTF-8 
import torch
# print(torch.__version__) 
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [3]))

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.cuda.is_available() 
# torch.cuda.device_count()  
# torch.cuda.current_device()

print(torch.cuda.get_device_name(CUDA_VISIBLE_DEVICES))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
import time
import collections

from shutil import copyfile

from data_utils import *
from evaluate import *
from distribution import *

dataset_base_path="/data/fan_xin/Philadelphia"

Div = "Cate"
# Div = "Geo"

user_num=4716
item_num=7727
# user_num=2419
# item_num=4964
factor_num=64
batch_size=1024
top_k=10
num_negative_test_val=-1##all  

sample_number=500

start_i_test=900
end_i_test=1000
setp=10

run_id="s6"
print(run_id)
dataset='Philadelphia'

path_save_model_base='/data/fan_xin/newlossModel_/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    pdb.set_trace() 

training_user_set = np.load(dataset_base_path+'/training_user_set.npy',allow_pickle=True).item()
training_item_set = np.load(dataset_base_path+'/training_item_set.npy',allow_pickle=True).item()
testing_user_set = np.load(dataset_base_path+'/testing_user_set.npy',allow_pickle=True).item()
testing_item_set = np.load(dataset_base_path+'/testing_item_set.npy',allow_pickle=True).item()
user_rating_set_all = np.load(dataset_base_path+'/user_rating_set_all.npy',allow_pickle=True).item()
training_set_count = Count(training_user_set)


def readD(set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d

u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)

#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1))
            user_items_matrix_v.append(d_i_j)#(1./len_set) 

    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i=readTrainSparseMatrix(training_user_set,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,False)


class BPR(nn.Module):
  def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix):
    super(BPR, self).__init__()
    """
    user_num: number of users;
    item_num: number of items;
    factor_num: number of predictive factors.
    """
    self.user_item_matrix = user_item_matrix
    self.item_user_matrix = item_user_matrix
    self.embed_user = nn.Embedding(user_num, factor_num)
    self.embed_item = nn.Embedding(item_num, factor_num)    

    nn.init.normal_(self.embed_user.weight, std=0.01)
    nn.init.normal_(self.embed_item.weight, std=0.01) 

    # self.d_i_train=d_i_train
    # self.d_j_train=d_j_train 

  def forward(self, user_item_3, item_user_3, user_js): 

    users_embedding=self.embed_user.weight
    items_embedding=self.embed_item.weight 
    
    gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding))  #users_embedding.mul(self.d_i_train)) #*2. #+ users_embedding
    gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding))  #items_embedding.mul(self.d_j_train)) #*2. #+ items_embedding
    
    gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding)) #gcn1_users_embedding.mul(self.d_i_train)) #*2. + users_embedding
    gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding)) #gcn1_items_embedding.mul(self.d_j_train)) #*2. + items_embedding

    gcn3_users_embedding = (torch.sparse.mm(user_item_3, items_embedding)) #gcn2_users_embedding.mul(self.d_i_train)) #*2. + gcn1_users_embedding
    gcn3_items_embedding = (torch.sparse.mm(user_item_3.t(), users_embedding)) #gcn2_items_embedding.mul(self.d_j_train)) #*2. + gcn1_items_embedding
    
    # gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(self.d_i_train)) #*2. + gcn1_users_embedding
    # gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(self.d_j_train)) #*2. + gcn1_items_embedding
    
    # gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding,gcn4_users_embedding),-1) #+gcn4_users_embedding
    # gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding,gcn4_items_embedding),-1) #+gcn4_items_embedding#

    gcn_users_embedding = (1/4)*users_embedding + (1/4)*gcn1_users_embedding + (1/4)*gcn2_users_embedding + torch.mul(gcn3_users_embedding, user_js)
    gcn_items_embedding = (1/4)*items_embedding + (1/4)*gcn1_items_embedding + (1/4)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

    # gcn_users_embedding = (1/4)*users_embedding + (1/4)*gcn1_users_embedding + (1/4)*gcn2_users_embedding + torch.mul(gcn3_users_embedding, user_js)
    # gcn_items_embedding = (1/4)*items_embedding + (1/4)*gcn1_items_embedding + (1/4)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

    return gcn_users_embedding, gcn_items_embedding


test_batch=52#int(batch_size/32)
testing_dataset = resData(train_dict=testing_user_set, batch_size=test_batch,num_item=item_num,all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0)

model = BPR(user_num, item_num, factor_num, sparse_u_i, sparse_i_u)
model=model.to('cuda')

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99))

user_candidate = np.load(dataset_base_path+'/user_candidate.npy', allow_pickle=True).item()

cluster_num = 0
offset = 0

if Div == "Geo":
    pass
else:
    cluster = np.load(dataset_base_path+'/cluster_.npy', allow_pickle=True).item()
    cluster_num  = cluster_number(cluster)
    user_first = first_layer_distribution_(training_user_set, cluster, cluster_num)

    user_third_temp = third_layer_distribution_without_sample(user_candidate, cluster, cluster_num)
    offset = calculate_offset(user_first, user_third_temp, user_num)

    user_candidate = specific_candidate_(user_first, user_candidate, cluster)

########################### TRAINING ##################################### 
# testing_loader_loss.dataset.ng_sample()

print('--------test processing-------')
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp):
    model.train()   
    
    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    #torch.save(model.state_dict(), PATH_model) 
    model.load_state_dict(torch.load(PATH_model)) 
    model.eval()     
    # ######test and val###########    

    if Div == "Geo":
        pass 
    else:
        user_third, user_sample = third_layer_distribution_(user_candidate, sample_number, cluster, cluster_num)

    # PATH_model = path_save_model_base+'/epoch_third'+str(epoch)+'.npy'
    # user_third = np.load(PATH_model, allow_pickle=True).item()
    # PATH_model = path_save_model_base+'/epoch_sample'+str(epoch)+'.npy'
    # user_sample = np.load(PATH_model, allow_pickle=True).item()

    user_js = [0]*user_num

    for u in training_user_set.keys():
      if not user_first[u] or not user_third[u]:
        continue
      user_js[u] = JS(user_first[u], user_third[u])

# 3.24 / 1.76 / 1.57

    user_js = torch.tensor([user_js]).t().cuda() * (1/(4*offset))

    user_sample[user_num-1].add(item_num-1)

    u_d=readD(user_sample,user_num)
    
    sparse_u_i=readTrainSparseMatrix(user_sample,True)

    # gcn_users_embedding, gcn_items_embedding,gcn_user_emb,gcn_item_emb = model()
    gcn_users_embedding, gcn_items_embedding = model(sparse_u_i, sparse_u_i.t(), user_js)
    user_e=gcn_users_embedding.cpu().detach().numpy()
    item_e=gcn_items_embedding.cpu().detach().numpy()
    all_pre=np.matmul(user_e,item_e.T)
    HR, NDCG, PRECISION, RECALL, ILD, COV, GS = [], [], [], [], [], [], []
    set_all=set(range(item_num)) 
    #spend 461s
    test_start_time = time.time()
    for u_i in testing_user_set:
    
        item_i_list = list(testing_user_set[u_i])
        index_end_i = len(item_i_list)
        item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list) 
        pre_one=all_pre[u_i][item_i_list]

        original = sorted(list(zip(pre_one, item_i_list)), reverse=True)[0:top_k]
        score, recList = zip(*original)
        
        gs_t = calculate_js(np.array(user_first[u_i])/sum(user_first[u_i]), recList, cluster)

        if Div == "Geo":
            cov_t = calculate_coverage(user_first[u_i], recList, cluster)
        else:
            cov_t = calculate_coverage_(user_first[u_i], recList, cluster)

        ild_t = ILD_geo(recList)    
        precision_t = precision(recList, list(testing_user_set[u_i]))
        recall_t = recall(recList, list(testing_user_set[u_i]))
        hr_t,ndcg_t = hr_ndcg_(recList, list(testing_user_set[u_i]), top_k)
        
        elapsed_time = time.time() - test_start_time

        HR.append(hr_t)
        NDCG.append(ndcg_t)
        PRECISION.append(precision_t)
        RECALL.append(recall_t)
        ILD.append(ild_t)
        COV.append(cov_t)
        GS.append(gs_t)

    hr_test = round(np.mean(HR),4)
    ndcg_test = round(np.mean(NDCG),4)
    precision_test = round(np.mean(PRECISION),4)
    ild_test = round(np.mean(ILD),4)
    cov_test = round(np.mean(COV),4)
    gs_test = round(np.mean(GS), 4)

    # test_loss,hr_test,ndcg_test = evaluate.metrics(model,testing_loader,top_k,num_negative_test_val,batch_size)  
    str_print_evl="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,2))+"\t test"+" hit:"+str(hr_test)+' ndcg:'+str(ndcg_test)+' precision:'+str(precision_test)+' ild:'+str(ild_test)+ ' coverage:'+str(cov_test)+' js:'+str(gs_test)
    print(str_print_evl)

