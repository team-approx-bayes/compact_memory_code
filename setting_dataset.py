import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from torchvision import datasets
from collections import OrderedDict
from time import time

import torch
import torch.nn as nn
import random
import numpy as np



## -- dataset --
from copy import deepcopy
from scipy import ndimage
from torch.utils.data import TensorDataset, Dataset, Subset
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms

import pickle
import gzip   

import torch.nn as nn
import os
 


from tqdm import tqdm
import clip



#username         ='abc'
#dataset_path     = '/home/{}/Datasets/'.format(username) 
dataset_path     = '~/Datasets/' 


def generate_setting_permnist(seed=111,ftype='cliprn50'):
    print('## choen featurep = {} ##'.format(ftype))
    

    #seed=111
    task_path='./dataset/permutedmnist_{}/'.format(ftype)
    os.makedirs(task_path,exist_ok=True)
    save_dict_name = 'task_seed{}.npz'.format(seed)

    num_workers = 4
    batch_size  = 1024
    image_osize = 28
    #max_iter    = 10            
    max_iter    = 5            

    n_class  = 10
    n_task   = max_iter             


    if os.path.exists(task_path +save_dict_name):
        print('## task loading ##')

        task_loaded = np.load(task_path +save_dict_name,allow_pickle=True)

        train_feature_dict = task_loaded['train_feature_dict'].item()
        train_labels_dict  = task_loaded['train_labels_dict'].item()
        test_feature_dict  = task_loaded['test_feature_dict'].item()
        test_labels_dict   = task_loaded['test_labels_dict'].item()


    else:
        from collections import OrderedDict
         
        train_inputs_dict = OrderedDict({})
        train_labels_dict = OrderedDict({})
        test_inputs_dict  = OrderedDict({})
        test_labels_dict  = OrderedDict({})
        
        train_feature_dict = OrderedDict({})
        test_feature_dict  = OrderedDict({})
        

         
        if ftype == 'cliprn50':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN50', device)
                    


        if ftype == 'clipvitb32':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-B/32', device)
                    

        if ftype == 'clipvitl14':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14', device)



        for cur_iter in range(max_iter):
            # set random seed 
            np.random.seed(cur_iter + seed)
            perm_inds = np.arange(image_osize**2)
            np.random.shuffle(perm_inds)
            

            full_dataset     = datasets.MNIST(dataset_path , train=True, download=True, transform=preprocess)
            full_dataset_tmp = full_dataset.data.reshape(full_dataset.data.size(0), -1)
            test_dataset     = datasets.MNIST(dataset_path , train=False, download=True, transform=preprocess)
            test_dataset_tmp = test_dataset.data.reshape(test_dataset.data.size(0), -1)


            ## permuted 
            if cur_iter >= 1:
                full_dataset_perm = (full_dataset_tmp[:,perm_inds]).reshape(full_dataset_tmp.size(0),image_osize,-1)
                test_dataset_perm = (test_dataset_tmp[:,perm_inds]).reshape(test_dataset_tmp.size(0),image_osize,-1)
                
                full_dataset.data = full_dataset_perm
                test_dataset.data = test_dataset_perm
                


            with torch.no_grad():
                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    #breakpoint()
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())

                train_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                train_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())
                        
                test_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                test_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)




            task_dict= {'train_feature_dict' :train_feature_dict,
                        'train_labels_dict'  :train_labels_dict,
                        'test_feature_dict'  :test_feature_dict,
                        'test_labels_dict'   :test_labels_dict,
            }


            np.savez(task_path +save_dict_name, **task_dict)



    task_inputs_list = [i_values  for i_values in train_feature_dict.values() ]
    task_labels_list = [i_values  for i_values in train_labels_dict.values() ]
    
    test_inputs_list = [i_values  for i_values in test_feature_dict.values() ]
    test_labels_list = [i_values  for i_values in test_labels_dict.values() ]



    return task_inputs_list,task_labels_list,test_inputs_list,test_labels_list,n_class,n_task







def generate_setting_splitcifar10(seed=111,ftype='cliprn50'):

    n_class  = 10
    n_task   = 5            


    ## reference for split cifar10, cifar 100, and TinyImagenet
    # https://github.com/visiontao/cl/blob/main/main.py#L87
    def _get_split(dataset, t, n_classes, n_tasks):    
        n_cpt     = n_classes // n_tasks  # n classes per task
        min_label = n_cpt * t
        max_label = n_cpt * (t + 1)
        
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] in range(min_label, max_label):
                indices.append(i)

        split = torch.utils.data.Subset(dataset, indices)
            
        return split


    #seed=111
    task_path='./dataset/splitcifar10_{}/'.format(ftype)
    os.makedirs(task_path,exist_ok=True)
    save_dict_name = 'task_seed{}.npz'.format(seed)



    num_workers = 4
    batch_size  = 1024



    

    if os.path.exists(task_path +save_dict_name):
        print('## task loading ##')

        task_loaded = np.load(task_path +save_dict_name,allow_pickle=True)

        train_feature_dict = task_loaded['train_feature_dict'].item()
        train_labels_dict  = task_loaded['train_labels_dict'].item()
        test_feature_dict  = task_loaded['test_feature_dict'].item()
        test_labels_dict   = task_loaded['test_labels_dict'].item()


    else:
        #get t-th split from the dataset for Class-IL and Task-IL


        import clip

        from collections import OrderedDict
            
        train_inputs_dict = OrderedDict({})
        train_labels_dict = OrderedDict({})
        test_inputs_dict  = OrderedDict({})
        test_labels_dict  = OrderedDict({})

        train_feature_dict = OrderedDict({})
        test_feature_dict  = OrderedDict({})



            
        if ftype == 'cliprn50':                

            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN50', device)
                    

            
        if ftype == 'clipvitb32':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-B/32', device)
                    
            
        if ftype == 'clipvitl14':
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14', device)


        for cur_iter in range(n_task):
            # set random seed 
            

            full_dataset        = datasets.CIFAR10(dataset_path , train=True, download=True, transform=preprocess)
            full_dataset_subset = _get_split(full_dataset, cur_iter, n_class, n_task) 
            test_dataset        = datasets.CIFAR10(dataset_path , train=False, download=True, transform=preprocess)
            test_dataset_subset = _get_split(test_dataset, cur_iter, n_class, n_task) 

                


            with torch.no_grad():
                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=full_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    #breakpoint()
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())

                train_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                train_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=test_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())
                        
                test_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                test_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



        task_dict= {'train_feature_dict' :train_feature_dict,
                    'train_labels_dict'  :train_labels_dict,
                    'test_feature_dict'  :test_feature_dict,
                    'test_labels_dict'   :test_labels_dict,
        }


        np.savez(task_path +save_dict_name, **task_dict)
        print('saved at ' +task_path +save_dict_name )





    task_inputs_list = [i_values  for i_values in train_feature_dict.values() ]
    task_labels_list = [i_values  for i_values in train_labels_dict.values() ]
    
    test_inputs_list = [i_values  for i_values in test_feature_dict.values() ]
    test_labels_list = [i_values  for i_values in test_labels_dict.values() ]


    return task_inputs_list,task_labels_list,test_inputs_list,test_labels_list,n_class,n_task







def generate_setting_splitcifar100(seed=111,ftype='cliprn50'):

    n_class  = 100
    n_task   = 10           


    ## reference for split cifar10, cifar 100, and TinyImagenet
    # https://github.com/visiontao/cl/blob/main/main.py#L87
    def _get_split(dataset, t, n_classes, n_tasks):    
        n_cpt     = n_classes // n_tasks  # n classes per task
        min_label = n_cpt * t
        max_label = n_cpt * (t + 1)
        
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] in range(min_label, max_label):
                indices.append(i)

        split = torch.utils.data.Subset(dataset, indices)
            
        return split


    #seed=111
    task_path='./dataset/splitcifar100_{}/'.format(ftype)
    os.makedirs(task_path,exist_ok=True)
    save_dict_name = 'task_seed{}.npz'.format(seed)



    num_workers = 4
    batch_size  = 1024



    if os.path.exists(task_path +save_dict_name):
        print('## task loading ##')

        task_loaded = np.load(task_path +save_dict_name,allow_pickle=True)

        train_feature_dict = task_loaded['train_feature_dict'].item()
        train_labels_dict  = task_loaded['train_labels_dict'].item()
        test_feature_dict  = task_loaded['test_feature_dict'].item()
        test_labels_dict   = task_loaded['test_labels_dict'].item()


    else:
        #get t-th split from the dataset for Class-IL and Task-IL


        import clip

        from collections import OrderedDict
            
        train_inputs_dict = OrderedDict({})
        train_labels_dict = OrderedDict({})
        test_inputs_dict  = OrderedDict({})
        test_labels_dict  = OrderedDict({})

        train_feature_dict = OrderedDict({})
        test_feature_dict  = OrderedDict({})



            
        if ftype == 'cliprn50':                

            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN50', device)
                    

            
        if ftype == 'clipvitb32':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-B/32', device)
                    
            
        if ftype == 'clipvitl14':
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14', device)


        for cur_iter in range(n_task):
            # set random seed 
            

            full_dataset        = datasets.CIFAR100(dataset_path , train=True, download=True, transform=preprocess)
            full_dataset_subset = _get_split(full_dataset, cur_iter, n_class, n_task) 
            test_dataset        = datasets.CIFAR100(dataset_path , train=False, download=True, transform=preprocess)
            test_dataset_subset = _get_split(test_dataset, cur_iter, n_class, n_task) 

                


            with torch.no_grad():
                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=full_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    #breakpoint()
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())

                train_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                train_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=test_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())
                        
                test_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                test_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



        task_dict= {'train_feature_dict' :train_feature_dict,
                    'train_labels_dict'  :train_labels_dict,
                    'test_feature_dict'  :test_feature_dict,
                    'test_labels_dict'   :test_labels_dict,
        }


        np.savez(task_path +save_dict_name, **task_dict)
        print('saved at ' +task_path +save_dict_name )





    task_inputs_list = [i_values  for i_values in train_feature_dict.values() ]
    task_labels_list = [i_values  for i_values in train_labels_dict.values() ]
    
    test_inputs_list = [i_values  for i_values in test_feature_dict.values() ]
    test_labels_list = [i_values  for i_values in test_labels_dict.values() ]


    return task_inputs_list,task_labels_list,test_inputs_list,test_labels_list,n_class,n_task










def generate_setting_splittinyimagenet(seed=111,ftype='cliprn50'):

    n_class  = 200
    n_task   = 20           


    ## reference for split cifar10, cifar 100, and TinyImagenet
    # https://github.com/visiontao/cl/blob/main/main.py#L87
    def _get_split(dataset, t, n_classes, n_tasks):    
        n_cpt     = n_classes // n_tasks  # n classes per task
        min_label = n_cpt * t
        max_label = n_cpt * (t + 1)
        
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] in range(min_label, max_label):
                indices.append(i)

        split = torch.utils.data.Subset(dataset, indices)
            
        return split


    #seed=111
    task_path='./dataset/tinyimagenet_{}/'.format(ftype)
    os.makedirs(task_path,exist_ok=True)
    save_dict_name = 'task_seed{}.npz'.format(seed)




    num_workers = 4
    batch_size  = 1024



    

    if os.path.exists(task_path +save_dict_name):
        print('## task loading ##')

        task_loaded = np.load(task_path +save_dict_name,allow_pickle=True)

        train_feature_dict = task_loaded['train_feature_dict'].item()
        train_labels_dict  = task_loaded['train_labels_dict'].item()
        test_feature_dict  = task_loaded['test_feature_dict'].item()
        test_labels_dict   = task_loaded['test_labels_dict'].item()


    else:
        #get t-th split from the dataset for Class-IL and Task-IL


        import clip

        from collections import OrderedDict
            
        train_inputs_dict = OrderedDict({})
        train_labels_dict = OrderedDict({})
        test_inputs_dict  = OrderedDict({})
        test_labels_dict  = OrderedDict({})

        train_feature_dict = OrderedDict({})
        test_feature_dict  = OrderedDict({})



            
        if ftype == 'cliprn50':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN50', device)
                    

            
        if ftype == 'clipvitb32':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-B/32', device)
                    
            
        if ftype == 'clipvitl14':
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14', device)


        if ftype == 'clipvitl14-336':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14@336px', device)



        # Set dataset path
        
        for cur_iter in range(n_task):
            # set random seed 
            



            # dataset_path : directory of tinyimagenet
            full_dataset        = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=preprocess)
            full_dataset_subset = _get_split(full_dataset, cur_iter, n_class, n_task) 
            test_dataset        = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=preprocess)
            test_dataset_subset = _get_split(test_dataset, cur_iter, n_class, n_task) 

                


            with torch.no_grad():
                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=full_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    #breakpoint()
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())

                train_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                train_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=test_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())
                        
                test_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                test_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



        task_dict= {'train_feature_dict' :train_feature_dict,
                    'train_labels_dict'  :train_labels_dict,
                    'test_feature_dict'  :test_feature_dict,
                    'test_labels_dict'   :test_labels_dict,
        }


        np.savez(task_path +save_dict_name, **task_dict)
        print('saved at ' +task_path +save_dict_name )





    task_inputs_list = [i_values  for i_values in train_feature_dict.values() ]
    task_labels_list = [i_values  for i_values in train_labels_dict.values() ]
    
    test_inputs_list = [i_values  for i_values in test_feature_dict.values() ]
    test_labels_list = [i_values  for i_values in test_labels_dict.values() ]


    return task_inputs_list,task_labels_list,test_inputs_list,test_labels_list,n_class,n_task








def generate_setting_splitimagenet(seed=111,ftype='cliprn50'):

    n_class  = 1000
    n_task   = 10           


    ## reference for split cifar10, cifar 100, and TinyImagenet
    # https://github.com/visiontao/cl/blob/main/main.py#L87
    def _get_split(dataset, t, n_classes, n_tasks):    
        n_cpt     = n_classes // n_tasks  # n classes per task
        min_label = n_cpt * t
        max_label = n_cpt * (t + 1)
        
        indices = []
        for i in range(len(dataset)):
            if dataset.targets[i] in range(min_label, max_label):
                indices.append(i)

        split = torch.utils.data.Subset(dataset, indices)
            
        return split


    #seed=111
    task_path='./dataset/imagenet_{}/'.format(ftype)
    os.makedirs(task_path,exist_ok=True)
    save_dict_name = 'task_seed{}.npz'.format(seed)




    num_workers = 4
    #batch_size  = 1024
    batch_size  = 2048


    

    if os.path.exists(task_path +save_dict_name):
        print('## task loading ##')

        task_loaded = np.load(task_path +save_dict_name,allow_pickle=True)

        train_feature_dict = task_loaded['train_feature_dict'].item()
        train_labels_dict  = task_loaded['train_labels_dict'].item()
        test_feature_dict  = task_loaded['test_feature_dict'].item()
        test_labels_dict   = task_loaded['test_labels_dict'].item()


    else:
        #get t-th split from the dataset for Class-IL and Task-IL


        import clip

        from collections import OrderedDict
            
        train_inputs_dict = OrderedDict({})
        train_labels_dict = OrderedDict({})
        test_inputs_dict  = OrderedDict({})
        test_labels_dict  = OrderedDict({})

        train_feature_dict = OrderedDict({})
        test_feature_dict  = OrderedDict({})



            
        if ftype == 'cliprn50':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('RN50', device)
                    

            
        if ftype == 'clipvitb32':                
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-B/32', device)
                    
            
        if ftype == 'clipvitl14':
            # Load the model
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14', device)


        if ftype == 'clipvitl14-336':
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load('ViT-L/14@336px', device)



        
        for cur_iter in tqdm( range(n_task) ,desc='progress bar'):
            # set random seed 
            
            # dataset_path : directory of imagenet
            
            full_dataset        = datasets.ImageFolder(os.path.join(dataset_path, "train"), transform=preprocess)
            full_dataset_subset = _get_split(full_dataset, cur_iter, n_class, n_task) 

            test_dataset        = datasets.ImageFolder(os.path.join(dataset_path, "val"), transform=preprocess)
            test_dataset_subset = _get_split(test_dataset, cur_iter, n_class, n_task) 

                


            with torch.no_grad():
                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=full_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):

                #for x,y in tqdm(DataLoader(dataset=full_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers),desc='progress bar'):
                    #breakpoint()
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())

                train_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                train_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



                tmp_feature_list,tmp_label_list = [],[]
                for x,y in DataLoader(dataset=test_dataset_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers):
                    
                    features              = model.encode_image(x.to(device))        
                    tmp_feature           = features.cpu().data.numpy() 
                    tmp_feature           = np.concatenate([tmp_feature, np.ones(tmp_feature.shape[0]).reshape(-1,1)],axis=-1)        
                    tmp_feature_list.append(tmp_feature)    
                    tmp_label_list.append(y.cpu().data.numpy())
                        
                test_feature_dict[cur_iter] = np.concatenate(tmp_feature_list,axis=0).astype(np.float64)
                test_labels_dict[cur_iter]  = np.concatenate(tmp_label_list,axis=0)



        task_dict= {'train_feature_dict' :train_feature_dict,
                    'train_labels_dict'  :train_labels_dict,
                    'test_feature_dict'  :test_feature_dict,
                    'test_labels_dict'   :test_labels_dict,
        }


        np.savez(task_path +save_dict_name, **task_dict)
        print('saved at ' +task_path +save_dict_name )





    task_inputs_list = [i_values  for i_values in train_feature_dict.values() ]
    task_labels_list = [i_values  for i_values in train_labels_dict.values() ]
    
    test_inputs_list = [i_values  for i_values in test_feature_dict.values() ]
    test_labels_list = [i_values  for i_values in test_labels_dict.values() ]


    return task_inputs_list,task_labels_list,test_inputs_list,test_labels_list,n_class,n_task
