from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from collections import OrderedDict
import pandas as pd
import numpy as np
import torch
import torchvision
from PIL.Image import LANCZOS
import sklearn.model_selection as model_selection

################################################
class Continual_Dataset(Dataset):
    def __init__(self, config, data_x, data_y):
        self.config = config
        self.x  = data_x
        self.y  = data_y

        if self.config['problem'] == 'classification':
            if self.config['network'] == 'fcnn': 
                self.x  =  self.x.reshape([-1,784])

    # A function to define the length of the problem
    def __len__(self):
        return self.x.shape[0]

    # A function to get samples
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.config['network'] == 'cnn':
            x_  =  self.x[idx, :,:,:]
            y_  =  self.y[idx]
        else:
            x_  =  self.x[idx,:]
            y_  =  self.y[idx]
        sample = {'x': x_, 'y': y_}
        return sample


class data_return():
    def __init__(self, Config):
        self.config = Config
        self.dataset_id = Config['data_id']
        self.dataset = None
        self.len_exp_replay = Config['len_exp_replay']

        if self.dataset_id == 'omni':
            self.dataset = torchvision.datasets.Omniglot(root="../data", download=True, transform=transforms.Compose([
                                                                transforms.Resize(28, interpolation=LANCZOS),
                                                                transforms.ToTensor(),
                                                                lambda x: 1.0 - x,
                                                            ])
                                                            )


        if self.dataset_id == 'cifar10':
            transform_train = transforms.Compose([
                transforms.Resize(28, interpolation=LANCZOS),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            self.dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            


        if self.dataset_id == 'mnist':
            self.dataset = torchvision.datasets.MNIST(root="../data", download=True, transform=torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor(),
                                torchvision.transforms.Normalize(
                                    (0.1307,), (0.3081,))
                                ]) )



        if self.dataset_id == 'mnist' or self.dataset_id=='cifar10' or self.dataset_id == 'omni':
            [self.images, self.labels] = [ list(t) for t in zip(*self.dataset)]
            self.images = torch.stack(self.images, dim = 0)
            self.labels = np.array(self.labels)


        self.y_test      = None
        self.X_test      = None
        self.y_train     = None
        self.X_train     = None

        self.exp_x_train = []
        self.exp_y_train = []
        self.exp_x_test  = []
        self.exp_y_test  = []
    
    
##############################################
    # This is the sine dataset function
    def sine(self, task_id):
        # New Task
        import pickle
        with open('/gpfs/jlse-fs0/users/kraghavan/Continual/Incremental_Sine.p', 'rb') as fp:
            data = pickle.load(fp)
        y, time, phase, amplitude, frequency = data['task'+str(task_id)]
        X = np.concatenate([phase, amplitude.reshape([-1,1]), frequency.reshape([-1,1]) ], axis = 1)
        self.X_train, self.X_test,  self.y_train,  self.y_test = \
        model_selection.train_test_split(X, y, test_size = 0.2)
        

    # This is the omniglot dataset function.
    def omni(self, task_id):
        idx = self.labels == task_id
        X = self.images[idx]
        y = self.labels[idx]
        # print(X.shape, y.shape) # Split the data
        self.X_train, self.X_test,  self.y_train,  self.y_test = \
            model_selection.train_test_split(X, y, test_size = 0.2)

##############################################


##############################################

    def append_to_experience(self, task_id):

        # Check if the arrays looks OK.
        if isinstance(self.X_train, np.ndarray):
               # print("how does this look")
               self.X_train = torch.from_numpy(self.X_train) 
               self.X_test  = torch.from_numpy(self.X_test) 


        if task_id  > 0:
            self.exp_x_test  = torch.cat( (self.exp_x_test,self.X_test), dim = 0)
            self.exp_x_train = torch.cat( (self.exp_x_train, self.X_train), dim = 0)


            self.exp_y_test =  np.concatenate( [self.exp_y_test, self.y_test],axis = 0)
            self.exp_y_train = np.concatenate([self.exp_y_train, self.y_train],axis = 0)

            # print("the experiance test shapes", self.exp_y_test.shape, self.exp_x_test.shape)
            # print(dat_y, y_train)
        else:
            self.exp_x_train.extend(self.X_train)
            self.exp_y_train.extend(self.y_train)
            self.exp_x_test.extend(self.X_test)
            self.exp_y_test.extend(self.y_test)
            
            # Convert the list into torch tensor
            # print("after extending", len(self.exp_x_train), len(self.exp_x_test))
            self.exp_x_train = torch.stack(self.exp_x_train, dim = 0)
            self.exp_y_train = np.array(self.exp_y_train)

            self.exp_x_test  = torch.stack(self.exp_x_test, dim = 0)
            self.exp_y_test  = np.array(self.exp_y_test)

        # Check for the length of the replay
        if len(self.exp_x_train) > self.config['len_exp_replay']: 
            index = np.random.randint(0, self.exp_x_train.shape[0], self.config['len_exp_replay'])
            self.exp_x_train = self.exp_x_train[index,:]
            self.exp_y_train = self.exp_y_train[index]

        if len(self.exp_x_test) > self.config['len_exp_replay']: 
            index = np.random.randint(0, self.exp_x_test.shape[0], self.config['len_exp_replay'])
            self.exp_x_test = self.exp_x_test[index, :]
            self.exp_y_test = self.exp_y_test[index]

            # print("I shrunk,", self.exp_y_train.shape, self.exp_x_test.shape)


    def retreive_data(self, task_id, phase):
        if phase == 'training':
            if task_id > 0:
                return (self.X_train, self.y_train), (self.exp_x_train, self.exp_y_train)
                # print("The shapes I am returning are", self.X_train.shape, self.exp_x_train.shape)
            else:
                return (self.X_train, self.y_train), (self.X_train, self.y_train)

        elif phase == 'testing':
            if task_id > 0:
                if self.config['data_id'] == 'omni':
                    index = np.random.randint(0,self.X_test.shape[0], 128)
                    return (self.X_test[index,:] , self.y_test[index]), (self.exp_x_test, self.exp_y_test)
                return (self.X_test, self.y_test), (self.exp_x_test, self.exp_y_test)
            else:
                if self.config['data_id'] == 'omni':
                    index = np.random.randint(0,self.X_test.shape[0], 128)
                    return (self.X_test[index,:] , self.y_test[index]), (self.X_test[index,:] , self.y_test[index])

                return (self.X_test, self.y_test), (self.X_test, self.y_test)




    def generate_dataset(self, task_id, batch_size, phase):
        # print(phase)
        if phase == 'training':
            if self.dataset_id == 'sine':
                self.sine(task_id)
            else: 
                self.omni(task_id)

        (x, y), (dat_x, dat_y) = self.retreive_data(task_id, phase)
        
        # print( np.unique(y), np.unique(dat_y) )
        # print(dat_y.shape, y.shape)
        # print(phase, "The data in", x.shape, y.shape, dat_x.shape, dat_y.shape)

        dataset_curr = Continual_Dataset(self.config, data_x = x, data_y = y)
        dataset_exp = Continual_Dataset(self.config, data_x = dat_x, data_y = dat_y)

        return DataLoader(dataset_curr, batch_size= self.config['batch_size'], shuffle=True, num_workers=4),\
               DataLoader(dataset_exp,  batch_size= self.config['batch_size'], shuffle=True, num_workers=4)

