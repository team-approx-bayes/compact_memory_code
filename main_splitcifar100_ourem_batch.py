import os
import torch
import copy 
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from torchvision import datasets
from collections import OrderedDict
from time import time


import torch.nn as nn
import random
import numpy as np

import torch.nn.init as init
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


class Multilabel_LogisticRegression(nn.Module):
    def __init__(self, num_features, dim_out=10, dropout=0.):
        super().__init__()
        self.num_features = num_features
        self.lin = torch.nn.Linear(self.num_features,dim_out, bias=False).double()
    
    def forward(self, xin):
        x = self.lin(xin)
        return x



costfunc_data = torch.nn.CrossEntropyLoss(reduction='none')
costfunc_reg  = torch.nn.CrossEntropyLoss(reduction='none')

def construct_Kprior(model_new, model_old, Z, Z_scale, delta):
    if Z != None:
        
        with torch.no_grad():
            model_old.eval()
            fz_old   = model_old(Z) 
            pred_old = fz_old.softmax(dim=-1) 

        fz_new      = model_new(Z)
        loss_memory = costfunc_reg(fz_new,pred_old)
        loss        = (Z_scale*loss_memory).sum()


        l2_reg      = torch.sum((model_new.lin.weight - model_old.lin.weight)**2)

        loss += 0.5* delta * l2_reg
    else:
        l2_reg = torch.sum((model_new.lin.weight)**2)
        loss = 0.5* delta* l2_reg
    return loss





class Seq_Kprior(nn.Module):
    
    def __init__(self, dim_features, dim_output, num_inducing, delta, jitter=1e-6, dropout=0.):
        super().__init__()
        self.dim_features = dim_features
        self.K = num_inducing
        self.delta = delta
        self.jitter = jitter
        self.dropout = nn.Dropout(dropout)
        
        self.model_old = Multilabel_LogisticRegression(dim_features,dim_output)
        self.model     = Multilabel_LogisticRegression(dim_features,dim_output)


    def filter_pred(self,f_new, task_idx):
        mask = torch.zeros_like(f_new) 
        mask[:,task_idx] += 1     
        f_new =  torch.where(mask.bool(), f_new, -torch.inf*torch.ones_like(f_new) )      
        return f_new


    def loss_theta(self, phi_new, label_new, memory_bank_list = [], task_idx= None):


        f_new      = self.model(phi_new)       
        loss_new   = costfunc_data(f_new,label_new).sum()

        if len(memory_bank_list) == 0:
            l2_reg = torch.sum((self.model.lin.weight)**2)
            kprior = 0.5*self.delta* l2_reg
        else:
            l2_reg      = torch.sum((self.model.lin.weight - self.model_old.lin.weight)**2)
            kprior      = 0.5* delta * l2_reg
            
            mem_feature,mem_weightsqrt = memory_bank_list                            
            with torch.no_grad():
                self.model_old.eval()
                fz_old   = self.model_old(mem_feature.to(phi_new.device)) 
                pred_old = fz_old.softmax(dim=-1) 

            fz_new        = self.model(mem_feature.to(phi_new.device))
            loss_funreg   = costfunc_reg(fz_new,pred_old)
            weight_factor = (mem_weightsqrt**2).to(loss_funreg.device)
            
            kprior       += (weight_factor*loss_funreg).sum()
         
        return loss_new , kprior



    def predict_with_filter(self, phi_new, task_idx=None):
        f_new      = self.model(phi_new)       
        if task_idx is not None:
            f_new  = self.filter_pred(f_new,task_idx)
        return f_new



def selectet_mem_lambda(model, X, size):
    with torch.no_grad():
        fx = model(X)
        hfx = torch.sigmoid(fx).detach()
        criterion = hfx* (1. - hfx) 
        criterion = criterion.squeeze(1)
        ind = torch.argsort(criterion)
        mem = X[ind][-size:] 
        
    print('beta of chosem mem')
    print(criterion[ind][-size:])
            
    return mem#, hfx_mem






def selectet_mem_with_label_random( X ,Y , size):
    label_unique    = np.unique(Y)
    nclass_per_task = len(label_unique)  
    ndata_per_task  = size//nclass_per_task

    mem_idx = []
    for i_idx in label_unique:
        i_mem_idx = np.where(Y == i_idx)[0]        
        i_mem_idx = np.random.permutation(i_mem_idx )[:ndata_per_task]
        
        mem_idx.append(i_mem_idx)
    mem_idx = np.concatenate(mem_idx)

    mem_chosen   = (X[mem_idx]).detach().clone()
    label_chosen = Y[mem_idx]

            
    return mem_chosen,label_chosen





def write_msg(current_msg,path):    
    with open(path, 'a') as f:
        f.write(current_msg + '\n')
    print(current_msg)
    return 






def learning_memory_tilde_repr(new_feature, previous_memory, initialization, sigma=0.05):
    new_feature = new_feature.T.cuda()
    initialization = initialization.T.cuda()
    


    if len(previous_memory) == 0:
        uk = torch.cat([new_feature], dim=-1) #[feature_dim, num_data]
    else:
        previous_memory = previous_memory.T.cuda()
        uk = torch.cat([new_feature, previous_memory], dim=-1) #[feature_dim, num_data]


    S = 1/sigma * initialization.T@initialization + torch.diag(torch.ones(initialization.shape[1]).cuda())  


    S_inv = torch.linalg.inv(S)
    m     = 1/sigma * (S_inv @ initialization.T)@uk #[num_memory, num_data]

    A = torch.einsum('ki,ji->kj', uk, m) ##[feature_dim, num_memory]
    B = torch.linalg.inv(S_inv + torch.einsum('ij,kj->ik', m, m) + 1e-6*torch.eye(S_inv.size(0)).to(S_inv.device) )  ##[num_memory, num_memory]
    initialization = A@B
    return initialization.T





if __name__ == "__main__":


    import argparse
    import copy
    import os


    parser = argparse.ArgumentParser()
    parser.add_argument('--ftype', type=str, default='cliprn50', choices=['cliprn50','clipvitb32','clipvitl14'], help='algo name  - default') 
    parser.add_argument('--method',      type=str, default='our', help='algo name - default') 
 
    # k-prior theta
    parser.add_argument('--lrtheta', type=float, default=1e-1, help='number of em epochs')
    parser.add_argument('--nthetaupdate', type=int, default=5000, help='number of em epochs')
    parser.add_argument('--delta',   type=float,   default=1e-2, help='number of')

    # k-prior mem
    parser.add_argument('--meminit',      type=str, default='lambda', help='labmda or random') 
    parser.add_argument('--nmem', type=int, default=100, help='number of training epochs')
    parser.add_argument('--nmemupdate', type=int,   default=1000, help='number of em epochs')

    # ppca
    parser.add_argument('--emnoise', type=float,   default=1e-4, help='noise in PPCA-EM: amlost default')

    # exp 
    parser.add_argument('--v',    type=int, default=1, help='version')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')


    args      = parser.parse_args()
    args_dict = vars(args)

    # CUDA_VISIBIE_DEVICES=0 python3 main_usps_basereplay_poly.py --lrtheta 1e-1 --nthetaupdate 100 --delta 1e-2 --nmem 1 --nmemupdate 50 --v 1 --seed 1111



    ## set hyperparameters    
    ftype, method = args.ftype, args.method    
    lr_theta, ntheta_update, ninducing_per_task, nmem_update, delta, em_noise, v = args.lrtheta, args.nthetaupdate, args.nmem, args.nmemupdate, args.delta, args.emnoise , args.v





    # set configs
    msg_configs = ''
    for i_key in args_dict:
        msg_configs += '{}:{} |'.format(i_key,args_dict[i_key])
    #msg_configs += ' = '


    # set seed
    seed=args.seed
    random.seed(seed)                          # Python random module
    np.random.seed(seed)                       # Numpy
    torch.manual_seed(seed)                    # CPU
    torch.cuda.manual_seed(seed)               # Current GPU
    torch.cuda.manual_seed_all(seed)           # All GPUs
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False     # Disable optimizations that introduce randomness

    taskname          = 'splitc100'
    path_result_param = './results/{}/'.format(taskname)
    path_result_txt   = './result_total_{}_v{}.txt'.format(taskname,v)
    path_log_txt      = './logger_v{}_{}_{}.txt'.format(v,taskname,method)
 
    os.makedirs(path_result_param, exist_ok=True)
    

    write_msg(msg_configs,path_log_txt)






    ## ------------------------------------------------------------------------------------------------------------------------------------------------------------- ## 
    ##Load task


    from setting_dataset import generate_setting_splitcifar100
    task_inputs_list,task_labels_list,test_inputs_list,test_labels_list,n_class,n_task = generate_setting_splitcifar100(seed=seed,ftype=ftype)





    ## set model

    dim_features = task_inputs_list[0].shape[-1]
    Kprior       = Seq_Kprior(dim_features=dim_features, dim_output=n_class, num_inducing=ninducing_per_task, delta=delta).cuda()





    phi_new_norm  = torch.tensor([])
    phi_new_norm_sum,phi_new_norm_cnt = 0,0
    memory_bank_inputs_list = []
    memory_bank_labels_list = []
    memory_bank_list        = []
    



    obslabels_list  = []
    
    for i in range(len(task_inputs_list)):    
        phi_new         = torch.from_numpy(task_inputs_list[i])   
        label_new       = torch.from_numpy(task_labels_list[i])
        itrainset       = TensorDataset(phi_new, label_new)
        
    
        total_size      = phi_new.size(0)
        batch_size      = 1024
        itrainloader    = DataLoader(dataset=itrainset, batch_size=batch_size  , shuffle=True, num_workers=4)        
        optimizer_theta = torch.optim.Adam(Kprior.model.parameters(), lr=lr_theta)    



        ## ----------- K-prior ----------- ##
        
        for j in range(ntheta_update):
            
            j_loss_list,j_loss_new_list,j_kprior_list = [],[],[]
            for i_x,i_y in itrainloader:
                
                i_x = i_x.cuda()
                i_y = i_y.cuda()



                #breakpoint()
                optimizer_theta.zero_grad()
                loss_new , kprior = Kprior.loss_theta(i_x,i_y, memory_bank_list = memory_bank_list )

                loss              = loss_new/batch_size + kprior/total_size            
                loss.backward()
                optimizer_theta.step()

                j_loss_list.append( loss.item() ) 
                j_loss_new_list.append( loss_new.item() ) 
                j_kprior_list.append(  kprior.item() ) 



            if j % 10 == 0:                
                msg_train = 'task no. = {}, step = {}, loss = {:.4f}, loss d = {:.4f}, loss k = {:.4f}'.format(i+1, j+1, 
                                                                                                               np.array(j_loss_list).mean(),
                                                                                                               np.array(j_loss_new_list).mean(),
                                                                                                               np.array(j_kprior_list).mean() )
                write_msg(msg_train,path_log_txt)





        ## ----------- EM algorithm ----------- ##
        if i < n_task - 1:

            ###compute new phi tilde            
            softmax_eps = 1e-4
            with torch.no_grad():

            
                if i == 0:
                    ## observation part ##    
                    # phinew
                    phinew_pred              = Kprior.model(phi_new.cuda()).softmax(dim=-1).clamp(min=softmax_eps,max=1-softmax_eps)
                    phinew_beta_sqrt         = (phinew_pred*(1 - phinew_pred)).sum(dim=-1,keepdim=True).sqrt()
                    phinew_tilde_per_cls     = phinew_beta_sqrt.to(phi_new.device)*(phi_new) # (ndata,nfeature)


                    # memold                
                    prev_mem_tilde_per_cls   = []


                    ## learnable part ##    
                    memnew_init_per_cls ,label_init   = selectet_mem_with_label_random(phi_new, label_new, ninducing_per_task)
                    memnew_wsqrt_per_cls     = torch.ones(memnew_init_per_cls.size(0),1)
                    
                        
                else:
            
            
                    ## observation part ##    
                    # phinew
                    phinew_pred              = Kprior.model(phi_new.cuda()).softmax(dim=-1).clamp(min=softmax_eps,max=1-softmax_eps).cpu() 
                    phinew_beta_sqrt         = (phinew_pred*(1 - phinew_pred)).sum(dim=-1,keepdim=True).sqrt()
                    phinew_tilde_per_cls     = phinew_beta_sqrt.to(phi_new.device)*(phi_new) # (ndata,nfeature)


                    # memold
                    prev_mem_pred_per_cls     = Kprior.model(prev_mem_init_per_cls.cuda()).softmax(dim=-1).clamp(min=softmax_eps,max=1-softmax_eps).cpu()                
                    prev_mem_beta_per_cls     = (prev_mem_pred_per_cls*(1-prev_mem_pred_per_cls)).sum(dim=-1,keepdim=True).sqrt()
                    prev_mem_tilde_per_cls    = prev_mem_wsqrt_per_cls * prev_mem_beta_per_cls * prev_mem_init_per_cls
                



                    ## learnable part ##    
                    
                    # meminit       
                    memnew_init_per_cls ,label_init   = selectet_mem_with_label_random(phi_new, label_new, ninducing_per_task)
                    memnew_wsqrt_per_cls              = torch.ones(memnew_init_per_cls.size(0),1)
                                        
                    
                    # merge old and newinit
                    memnew_init_per_cls     = torch.cat([prev_mem_init_per_cls.cpu(),memnew_init_per_cls],dim=0)  #(ncls,nprevmeme +ndatanew,nfeature)
                    memnew_wsqrt_per_cls    = torch.cat([prev_mem_wsqrt_per_cls.cpu(),memnew_wsqrt_per_cls ],dim=0) #(ncls,nprevmeme +ndatanew)
                                    
                    
        
                
                
                for step_j in range(nmem_update):
                    

                    memnew_pred_per_cls            = Kprior.model(memnew_init_per_cls.cuda()).softmax(dim=-1).clamp(min=softmax_eps,max=1-softmax_eps).cpu() 
                    memnew_beta_per_cls            = (memnew_pred_per_cls*(1-memnew_pred_per_cls)).sum(dim=-1,keepdim=True).sqrt()
                    memnew_tilde_per_cls           = memnew_wsqrt_per_cls*memnew_beta_per_cls*memnew_init_per_cls
                    

                
                    memnew_tilde_update            = learning_memory_tilde_repr(phinew_tilde_per_cls , prev_mem_tilde_per_cls  , memnew_tilde_per_cls,  sigma=em_noise).cpu()     
 


                    ## re-transform
                    memnew_tilde_tmp                = memnew_tilde_update/memnew_beta_per_cls  
                    memnew_wsqrt_per_cls            = memnew_tilde_tmp.norm(dim=-1,keepdim=True) #(ndata,ncls)
                    memnew_init_per_cls             = memnew_tilde_tmp/(memnew_wsqrt_per_cls)



       
                    
                
                prev_mem_init_per_cls  = memnew_init_per_cls.detach().clone()
                prev_mem_wsqrt_per_cls = memnew_wsqrt_per_cls.detach().clone()                
                memory_bank_list       = ( prev_mem_init_per_cls , prev_mem_wsqrt_per_cls.squeeze() )
                

        else:            
            pass
                
                

        model_path = path_result_param + 'v{}_ftype{}_{}_kpdelta{}_nthetaup{}_nmem{}_nmemup{}_seqidx{}.pth.tar'.format(v, args.ftype,args.method,delta,ntheta_update,ninducing_per_task,nmem_update, i) 
        torch.save(Kprior.state_dict(), model_path)    
        Kprior.model_old = copy.deepcopy(Kprior.model)

        
        torch.cuda.empty_cache()





    ##evaluation
    Kprior_trained = Seq_Kprior(dim_features=dim_features, dim_output=n_class, num_inducing=ninducing_per_task, delta=delta).cuda()
    saved_param    = torch.load(model_path)
    Kprior_trained.load_state_dict(saved_param,strict=False)

    metric_dict       = OrderedDict({})
    total_correct_cnt = 0
    total_cnt         = 0

    with torch.no_grad():

        for i,(inputs,labels) in enumerate(zip(test_inputs_list,test_labels_list)):
            
            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)

            pred             = Kprior_trained.model(inputs.cuda()).softmax(dim=-1)
            pred_class       = pred.max(dim=-1)[1]            
            each_correct_cnt = (pred_class.squeeze() ==  labels.cuda()).sum()
            each_total_cnt   = len(labels)
            
            total_correct_cnt += each_correct_cnt
            total_cnt         += each_total_cnt

            #print(correct_cnt/total_cnt)
            metric_dict['t-{}'.format(i)] = each_correct_cnt/each_total_cnt

        metric_dict['t-avg'] = total_correct_cnt/total_cnt  


    # set configs
    msg_results = ''
    for i_key in args_dict:
        msg_results += '{}:{} |'.format(i_key,args_dict[i_key])
    msg_results += ' --> '

    for i_key in metric_dict:
        msg_results += ' {}:{:.3f},'.format(i_key,metric_dict[i_key])
    msg_results=msg_results[:-1]


    write_msg(msg_results,path_result_txt)
    write_msg('\n'*2,path_log_txt)


