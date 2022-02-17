import torch.nn as nn
from torch.autograd import Variable, grad
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd.profiler as profiler
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import time, copy
import gc
import torch
from core.dataloaders import *
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score


################################################
# Sanity Check  and initialize the CPU/GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")

# The class for defining the network architecture and the optimizer
class Net(torch.nn.Module):
    def __init__(self, Config):
        super(Net, self).__init__()
        self.config = Config
        if self.config['network']== 'fcnn':
            # Model h
            self.model_F = torch.nn.Sequential(
                torch.nn.Linear(self.config['D_in'], self.config['H']),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config['H'], self.config['H']),
                torch.nn.ReLU(),
                torch.nn.Linear(self.config['H'], self.config['D_in'])
            )

            # The g model and the buffer model are the same
            # Model g
            self.model_P = torch.nn.Sequential(
            torch.nn.Linear(self.config['D_in'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_out']) 
            )


            # Model buffer
            self.model_buffer = torch.nn.Sequential(
            torch.nn.Linear(self.config['D_in'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_out']) 
            )
            
        elif self.config['network']== 'cnn':

            # # The data 
            self.model_F = torch.nn.Sequential( 
            torch.nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            )


            self.model_P = torch.nn.Sequential( 
            torch.nn.Linear(7 * 7 * 64, self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'],  self.config['D_out'])
            )

            self.model_buffer = torch.nn.Sequential( 

            torch.nn.Linear(7 * 7 * 64, self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'],  self.config['D_out'])

            )

        elif self.config['network']== 'cnn3':
           self.model_F = torch.nn.Sequential( 
            torch.nn.Conv2d(3, 6, 5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(6, 16, 5),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Dropout()
            )


           self.model_P = torch.nn.Sequential( 
            torch.nn.Linear(256, self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_out'])
            )


           self.model_buffer = torch.nn.Sequential( 
            torch.nn.Linear(256, self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['H']),
            torch.nn.ReLU(),
            torch.nn.Linear(self.config['H'], self.config['D_out'])
            )

        self.optimizer  = torch.optim.Adam(  list(self.model_P.parameters()) + list(self.model_F.parameters()), lr = self.config['learning_rate'] )

    # Return the current score.
    def return_score(self, dataloader_eval_curr, dataloader_eval_exp):
        return self.evaluate_model(dataloader_eval_curr), self.evaluate_model(dataloader_eval_exp)


    # The function to get the outputs
    def evaluate_model(self, test_loader):
        self.model_P.eval()
        self.model_F.eval()
        test_loss = 0.0
        correct = 0.0
        for sample in test_loader:
            dat = sample['x'].float().to(device)
            if self.config['problem'] == 'classification': 
                feature_out = self.model_F(dat)
                output = F.log_softmax(self.model_P(feature_out.reshape(feature_out.size(0), -1) ) )
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(sample['y'].long().reshape([-1]).to(device).data.view_as(pred)).sum()
            else:
                feature_out = self.model_F(dat)
                output = self.model_P(feature_out.reshape(feature_out.size(0), -1) ) 
                test_loss += torch.nn.MSELoss()(output, sample['y'].float().to(device)).item()
        if self.config['problem'] == 'classification':  
            return (100. * correct / len(test_loader.dataset))
        else:
            return (test_loss / len(test_loader.dataset))

    def update_para(self, sample, optimizer):
        ################################
        dat = sample['x'].float().to(device)
        target = sample['y'].to(device)
        self.optimizer.zero_grad()
        if self.config['problem'] == 'classification':
            target = target.reshape([-1]).long()
            feature_out = self.model_F(dat)

            # print(feature_out.shape)
            y_pred = F.log_softmax(self.model_P(feature_out.reshape(feature_out.size(0), -1) ) )
            loss   = F.nll_loss(y_pred, target)
        else:
            y_pred = self.model_P( self.model_F( dat ) )
            loss   = torch.nn.MSELoss()(y_pred, target.float() )
        loss.backward(retain_graph = True)
        optimizer.step()
        return loss


    def ANML(self, x, y, ANML_phase):
        if ANML_phase == 'training_ANML_CML_1':
            x, y = x.to(device), y.to(device)
            feature_out = self.model_NLM(x)
            if self.config['network']=='cnn' or self.config['network']=='cnn3':
                feature_out = feature_out.reshape(feature_out.size(0), -1)
            y_pred = self.model_P(feature_out)
            J_k = self.config['criterion'](y_pred, y.squeeze_())
            self.optimizer.zero_grad()
            J_k.backward()
            self.optimizer_NLM.step()
            self.optimizer_NLM.step()
            del J_k, x, y, y_pred
        else:
            x, y = x.to(device), y.to(device)
            feature_out_NLM = self.model_NLM(x)
            feature_out = self.model_F(x)

            if self.config['network']=='cnn' or self.config['network']=='cnn3':
                feature_out = feature_out.reshape(feature_out.size(0), -1)
                feature_out_NLM = feature_out_NLM.reshape(feature_out_NLM.size(0), -1)

            feature_out = torch.mul(feature_out, feature_out_NLM)
            y_pred = self.model_P(feature_out)
            J_k = self.config['criterion'](y_pred, y.squeeze_())
            self.optimizer.zero_grad()
            J_k.backward()
            self.optimizer.step()
            self.optimizer.step()
            del J_k, x, y, y_pred


###########################################################################
    def CML(self, x, y, CML_phase):
        if CML_phase == 'training_ANML_CML_1':
            x, y = x.to(device), y.to(device)
            feature_out = self.model_F(x)
            if self.config['network']=='cnn' or self.config['network']=='cnn3':
                feature_out = feature_out.reshape(feature_out.size(0), -1)
            y_pred = self.model_buffer(feature_out)
            J_k = self.config['criterion'](y_pred, y.squeeze_())
            self.optimizer.zero_grad()
            J_k.backward()
            self.opt_buffer.step()
            self.opt_buffer.step()
            del J_k, x, y, y_pred
        else:
            x, y = x.to(device), y.to(device)
            feature_out = self.model_F(x)
            if self.config['network']=='cnn' or self.config['network']=='cnn3':
                feature_out = feature_out.reshape(feature_out.size(0), -1)
            y_pred = self.model_P(feature_out)
            J_k = self.config['criterion'](y_pred, y.squeeze_())
            self.optimizer.zero_grad()
            J_k.backward()
            self.optimizer.step()
            self.optimizer.step()
            del J_k, x, y, y_pred




###########################################################################
    def ER(self, dataloader_exp):
        exp_it = iter(dataloader_exp)
        for epoch in range(self.config['N']):
            try:
                sample = next(exp_it) 
            except StopIteration:
                # StopIteration is thrown if dataset ends
                # reinitialize data loader 
                exp_it = iter(dataloader_exp)
                sample  = next(exp_it) 
            self.update_para(sample, self.optimizer)
        return self


###########################################################################    
    def OML(self, dataloader_exp, dataloader_curr):
        exp_it = iter(dataloader_exp)
        curr_it = iter(dataloader_curr)
        for epoch in range(self.config['N_meta']): 
                try:
                    sample = next(curr_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    curr_it = iter(dataloader_curr)
                    sample  = next(curr_it) 
                self.update_para(sample, self.optimizer)

                for epoch in range(self.config['N_grad']):
                    # print("I am converting to float")
                    try:
                        sample = next(exp_it) 
                    except StopIteration:
                        # StopIteration is thrown if dataset ends
                        # reinitialize data loader 
                        exp_it = iter(dataloader_exp)
                        sample = next(exp_it)
                    self.update_para(sample, self.optimizer)
        return self


###########################################################################
    def backward(self, dataloader_curr, dataloader_exp, samp_num, phase = None):

        if self.config['opt'] == 'Naive':
            return self.ER(dataloader_curr)

        if self.config['opt'] == 'ER':
            return self.ER(dataloader_exp)
        
        if self.config['opt'] == 'OML':
            return self.OML(dataloader_exp, dataloader_curr)
            
        if self.config['opt'] == 'DPMCL':
            exp_it  = iter(dataloader_exp)
            curr_it = iter(dataloader_curr)
            for epoch in range(self.config['kappa']):
                ## Generalization to new task
                try:
                    sample_c = next(curr_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    curr_it = iter(dataloader_curr)
                    sample_c  = next(curr_it) 
                # self.update_para(sample_c, self.optimizer)

                ## Compensate for forgetting.
                try:
                    sample_e = next(exp_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    exp_it = iter(dataloader_exp)
                    sample_e  = next(exp_it) 
                l = self.update_para(sample_e, self.optimizer)

                # Compensate for the third term
                if self.config['zeta']>0:
                    # self.update_third_term(sample_e, sample_c)
                    #####################################################
                    ### Compensate for the Third term
                    index = np.random.randint(0,\
                    sample_c['x'].shape[0], \
                    self.config['n_new_points'])
                    
                    if self.config['problem'] == 'classification':
                        x_PN = torch.cat((sample_e['x'], sample_c['x'][index,:]), dim = 0).float().to(device)
                        y_PN = torch.cat((sample_e['y'], sample_c['y'][index]), dim = 0).to(device)
                        target = y_PN.reshape([-1]).long()
                        feature_out = self.model_F(x_PN)
                        feature_out = feature_out.reshape(feature_out.size(0), -1)
                    else:
                        x_PN = torch.cat((sample_e['x'], sample_c['x'][index,:]), dim = 0).float().to(device)
                        y_PN = torch.cat((sample_e['y'], sample_c['y'][index,:]), dim = 0).to(device)
                        feature_out = self.model_F( x_PN)



                    self.model_buffer.load_state_dict(self.model_P.state_dict())
                    self.opt_buffer = torch.optim.Adam( \
                        list(self.model_buffer.parameters()),\
                        lr = 0.01*self.config['learning_rate'] )
                    
                    for epoch in range(self.config['zeta']): 
                        self.opt_buffer.zero_grad()
                        if self.config['problem'] == 'classification':
                            y_pred = F.log_softmax( self.model_buffer(feature_out) )
                            loss_BUF  = F.nll_loss(y_pred, target)
                        else:
                            y_pred = self.model_buffer(feature_out)
                            loss_BUF   = torch.nn.MSELoss()(y_pred, y_PN.float() )
                        loss_BUF.backward(create_graph = True)
                        self.opt_buffer.step() 
                    
                    # Loss k+1
                    if self.config['problem'] == 'classification':
                        y_pred = F.log_softmax( self.model_buffer(feature_out) )
                        Loss_PN_1  = F.nll_loss(y_pred, target)
                    else:
                        y_pred     = self.model_buffer(feature_out )
                        Loss_PN_1  = torch.nn.MSELoss()(y_pred, y_PN.float() )
        
                    # Loss k
                    if self.config['problem'] == 'classification':
                        y_pred = F.log_softmax( self.model_P(feature_out) )
                        Loss_PN  = F.nll_loss(y_pred, target)
                    else:
                        y_pred     = self.model_P( feature_out )
                        Loss_PN  = torch.nn.MSELoss()(y_pred, y_PN.float() )

                    self.optimizer.zero_grad()
                    Total_L = self.config['eta']*Loss_PN \
                        + self.config['gamma']*(Loss_PN - Loss_PN_1) 
                    Total_L.backward(create_graph= True)
                    self.optimizer.step()
                    del self.opt_buffer
            
            return self

            # return self.DPMCL(dataloader_exp, dataloader_curr)
           

        if self.config['opt'] == 'CML':
            exp_it  = iter(dataloader_exp)
            curr_it = iter(dataloader_curr)
            # print("I am in this optimization")
            # Internal loop with samples across
            for _ in range( int(self.config['N']/2) ): 
                try:
                    sample = next(curr_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    curr_it = iter(dataloader_curr)
                    sample = next(curr_it) 


                x = sample['x'].float()
                if self.config['problem'] == 'classification':
                    y = sample['y'].long()
                else:
                    y = sample['y'].float()
                ################################
                try:
                    sample = next(exp_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    exp_it = iter(dataloader_exp)
                    sample = next(exp_it) 
                ########################
                ex = sample['x'].float()
                if self.config['problem'] == 'classification':
                    ey = sample['y'].long()
                else:
                    ey = sample['y'].float()
                x = torch.cat((x, ex))
                y = torch.cat((y, ey))
                sample = {}
                sample['x'] = x
                sample['y'] = y
                self.CML(x, y, 'training_ANML_CML_1')
            
            for _ in range( int(self.config['N']/2) ): 
                try:
                    sample = next(curr_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    curr_it = iter(dataloader_curr)
                    sample = next(curr_it) 


                x = sample['x'].float()
                if self.config['problem'] == 'classification':
                    y = sample['y'].long()
                else:
                    y = sample['y'].float()
                ################################
                try:
                    sample = next(exp_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    exp_it = iter(dataloader_exp)
                    sample = next(exp_it) 
                ########################
                ex = sample['x'].float()
                if self.config['problem'] == 'classification':
                    ey = sample['y'].long()
                else:
                    ey = sample['y'].float()
                x = torch.cat((x, ex))
                y = torch.cat((y, ey))
                sample = {}
                sample['x'] = x
                sample['y'] = y
                self.CML(x, y, 'training_ANML_CML_2') 
            
            return self
        
        if self.config['opt'] == 'ANML':
            exp_it  = iter(dataloader_exp)
            curr_it = iter(dataloader_curr)
            # print("I am in this optimization")
            # Internal loop with samples across
            for _ in range( int(self.config['N']/2) ): 
                try:
                    sample = next(curr_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    curr_it = iter(dataloader_curr)
                    sample = next(curr_it) 
                x = sample['x'].float()
                if self.config['problem'] == 'classification':
                    y = sample['y'].long()
                else:
                    y = sample['y'].float()
                ################################
                try:
                    sample = next(exp_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    exp_it = iter(dataloader_exp)
                    sample = next(exp_it) 
                ########################
                ex = sample['x'].float()
                if self.config['problem'] == 'classification':
                    ey = sample['y'].long()
                else:
                    ey = sample['y'].float()
                x = torch.cat((x, ex))
                y = torch.cat((y, ey))
                sample = {}
                sample['x'] = x
                sample['y'] = y
                self.ANML(x, y, 'training_ANML_CML_1')
            
            for _ in range( int(self.config['N']/2) ): 
                try:
                    sample = next(curr_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    curr_it = iter(dataloader_curr)
                    sample = next(curr_it) 


                x = sample['x'].float()
                if self.config['problem'] == 'classification':
                    y = sample['y'].long()
                else:
                    y = sample['y'].float()
                ################################
                try:
                    sample = next(exp_it) 
                except StopIteration:
                    # StopIteration is thrown if dataset ends
                    # reinitialize data loader 
                    exp_it = iter(dataloader_exp)
                    sample = next(exp_it) 
                ########################
                ex = sample['x'].float()
                if self.config['problem'] == 'classification':
                    ey = sample['y'].long()
                else:
                    ey = sample['y'].float()
                x = torch.cat((x, ex))
                y = torch.cat((y, ey))
                sample = {}
                sample['x'] = x
                sample['y'] = y
                self.ANML(x, y, 'training_ANML_CML_2') 
            
            return self

        
    