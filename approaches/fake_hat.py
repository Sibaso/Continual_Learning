import sys, time, os
import math

from networks.mlp_fake_hat import RDLinear
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
import csv
from networks.power_spherical import PowerSpherical
from networks.kl_divergence import KL_vMF_kappa_full, KL_Powerspherical
from torch.distributions.kl import kl_divergence
from torch.distributions import Beta, Normal, LogNormal


class Appr(object):

    def __init__(self,model,data_name,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=100,optim='Adam'):
        self.model=model.to(device)
        self.model_old = deepcopy(self.get_model(self.model))
        self.data_name = data_name
        self.file_name = 'direction_ucl_{}_{}'.format(self.model.name, data_name)

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.optim = optim
        self.alpha = 0.01
        self.beta = 0.03
        self.saved = 0


        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        if self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def get_model(self, model):
        model_dict = {}
        for n, m in model.named_modules():
            if isinstance(m, RDLinear):
                model_dict[n] = m.state_dict()
        return model_dict

    def train(self, t, xtrain, ytrain, xvalid, yvalid):

        # tasks = set(ytrain.cpu().numpy())
        # for t in tasks:
        #     self.model.add_task(t)

        # ytrain = torch.stack([(ytrain==t)*1.0 for t in self.model.tasks], dim=-1)
        # ytrain = torch.argmax(ytrain, dim=-1).long()
        # yvalid = torch.stack([(yvalid==t)*1.0 for t in self.model.tasks], dim=-1)
        # yvalid = torch.argmax(yvalid, dim=-1).long()

        self.model.to(device)
        best_acc = -np.inf
        lr = self.lr
        patience = self.lr_patience
        self.optimizer = self._get_optimizer(lr)
        with open('./results/{}_task_{}.csv'.format(self.file_name, t), mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['train loss', 'train acc', 'valid loss', 'valid acc'])
        
        # Loop epochs
        # try:
        for e in range(self.nepochs):
            # Train
            clock0=time.time()
            
            num_batch = xtrain.size(0)
            
            self.train_epoch(xtrain,ytrain)
            
            clock1=time.time()
            train_loss,train_acc=self.eval(xtrain,ytrain)
            clock2=time.time()
            print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.2f}% |'.format(
                e+1,1000*(clock1-clock0),
                1000*(clock2-clock1),train_loss,100*train_acc),end='')
            # Valid
            valid_loss,valid_acc=self.eval(xvalid,yvalid)
            print(' Valid: loss={:.3f}, acc={:5.2f}% |'.format(valid_loss,100*valid_acc),end='')
            
            # Adapt lr
            if valid_acc > best_acc:
                best_acc = valid_acc
                torch.save(self.model.state_dict(),'./trained_model/{}_task_{}.model'.format(self.file_name, t))
                patience = self.lr_patience
                print(' *', end='')
            
            else:
                patience -= 1
                if patience <= 0 and lr >self.lr_min:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                        
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()
            with open('./results/{}_task_{}.csv'.format(self.file_name, t), mode='a', newline='') as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([train_loss, train_acc, valid_loss, valid_acc])

        # Restore best
        self.model.load_state_dict(torch.load('./trained_model/{}_task_{}.model'.format(self.file_name, t)))
        self.model_old = deepcopy(self.get_model(self.model))
        self.saved = 1
        

    def train_epoch(self,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)
            targets=y[b].to(device)

            mini_batch_size = len(targets)
            # Forward current model
            outputs = self.model.forward(images, sample=True)

            xent = self.ce(outputs,targets)
            # loss = self.custom_regularization(self.model_old, self.get_model(self.model), mini_batch_size, xent)
            kld = self.kl_divergence(self.model_old, self.get_model(self.model))
            loss = xent + kld
            loss = loss / mini_batch_size
            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # for m in self.model.modules():
            #     if hasattr(m, 'gradient_correction'):
            #         m.gradient_correction(xent)

            if self.optim == 'SGD':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)

            self.optimizer.step()

            for m in self.model.modules():
                if hasattr(m, 'parameter_adjustment'):
                    m.parameter_adjustment()


    def eval(self,x,y):
        total_loss=0
        total_acc=0
        total_num=0
        self.model.eval()

        r = np.arange(x.size(0))
        r = torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            images=x[b].to(device)
            targets=y[b].to(device)
            
            # Forward
            outputs = self.model.forward(images)
                
            loss=self.ce(outputs,targets)
            values,indices=outputs.max(1)
            hits=(indices==targets).float()#*(values>0.5).float()

            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num


# custom regularization

    def kl_divergence(self, saver_net, trainer_net):
        kld = 0
        prev_weight_strength = nn.Parameter(torch.Tensor(28*28,1).uniform_(0,0)).cuda()
        alpha = self.alpha
        if self.saved:
            alpha = 1
        for (saver_name, saver_layer), (trainer_name, trainer_layer) in zip(saver_net.items(), trainer_net.items()):

            trainer_dir_loc = trainer_layer['dir_loc']
            trainer_dir_concentration = F.softplus(trainer_layer['dir_softplus_inv_concentration'])
            trainer_rad_mu = trainer_layer['rad_mu']
            trainer_rad_sigma = F.softplus(trainer_layer['rad_rho'])
            trainer_bias = trainer_layer['bias']

            saver_dir_loc = saver_layer['dir_loc']
            saver_dir_concentration = F.softplus(saver_layer['dir_softplus_inv_concentration'])
            saver_rad_mu = saver_layer['rad_mu']
            saver_rad_sigma = F.softplus(saver_layer['rad_rho'])
            saver_bias = saver_layer['bias']

            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_dir_loc) 
            concentration_init = ml_kappa(dim=fan_in, eps=self.model.eps)
            
            if 'fc' in trainer_name:
                std_init = math.sqrt((2 / fan_in) * self.model.ratio)
            if 'conv' in trainer_name:
                std_init = math.sqrt((2 / fan_out) * self.model.ratio)
 
            out_features, in_features = saver_dir_loc.shape
            saver_weight_strength = (std_init / saver_rad_sigma)
            curr_strength = saver_weight_strength.expand(out_features,in_features)
            prev_strength = prev_weight_strength.permute(1,0).expand(out_features,in_features)
            L2_strength = torch.max(curr_strength, prev_strength)

            prev_weight_strength = saver_weight_strength

            dir_loc_reg = ((L2_strength * trainer_dir_loc * saver_dir_loc) / (trainer_dir_loc.norm(2, dim=-1) * saver_dir_loc.norm(2, dim=-1)).unsqueeze(-1)).sum()

            q_dir = PowerSpherical(trainer_dir_loc, trainer_dir_concentration)
            p_dir = PowerSpherical(saver_dir_loc, saver_dir_concentration)
            kld_dir = KL_Powerspherical(q_dir, p_dir)

            q_rad = LogNormal(trainer_rad_mu, trainer_rad_sigma)
            p_rad = LogNormal(saver_rad_mu, saver_rad_sigma)
            kld_rad = kl_divergence(q_rad, p_rad)

            mu_bias_reg = ((trainer_bias-saver_bias) / saver_rad_sigma.squeeze()).norm(2)**2

            kld += kld_dir.sum() + 100 * kld_rad.sum() + 100 * mu_bias_reg + 100 * dir_loc_reg

        return kld



    def custom_regularization(self, saver_net, trainer_net, mini_batch_size, loss=None):
        
        dir_loc_reg_sum = mu_bias_reg_sum = rad_mu_reg_sum = 0
        L1_rad_mu_reg_sum = L1_mu_bias_reg_sum = 0
        rad_sigma_reg_sum = rad_sigma_normal_reg_sum = 0
        
        out_features_max = 512
        alpha = self.alpha
        if self.saved:
            alpha = 1
        
        if 'conv' in self.model_name:
            if self.data_name == 'omniglot':
                prev_weight_strength = nn.Parameter(torch.Tensor(1,1,1,1).uniform_(0,0)).cuda()
            elif self.data_name == 'cifa':
                prev_weight_strength = nn.Parameter(torch.Tensor(3,1,1,1).uniform_(0,0)).cuda()
        else:
            prev_weight_strength = nn.Parameter(torch.Tensor(28*28,1).uniform_(0,0)).cuda()

        
        for (saver_name, saver_layer), (trainer_name, trainer_layer) in zip(saver_net.items(), trainer_net.items()):
            # calculate mu regularization
            trainer_dir_loc = trainer_layer['dir_loc']
            trainer_dir_concentration = F.softplus(trainer_layer['dir_softplus_inv_concentration'])
            # trainer_dir_loc = trainer_layer['dir_rsampler.loc']
            # trainer_dir_concentration = F.softplus(trainer_layer['dir_rsampler.softplus_inv_concentration'])
            trainer_rad_mu = trainer_layer['rad_mu']
            trainer_rad_sigma = F.softplus(trainer_layer['rad_rho'])
            trainer_bias = trainer_layer['bias']

            saver_dir_loc = saver_layer['dir_loc']
            saver_dir_concentration = F.softplus(saver_layer['dir_softplus_inv_concentration'])
            # saver_dir_loc = saver_layer['dir_rsampler.loc']
            # saver_dir_concentration = F.softplus(saver_layer['dir_rsampler.softplus_inv_concentration'])
            saver_rad_mu = saver_layer['rad_mu']
            saver_rad_sigma = F.softplus(saver_layer['rad_rho'])
            saver_bias = saver_layer['bias']
            
            fan_in, fan_out = _calculate_fan_in_and_fan_out(trainer_dir_loc)
            
            concentration_init = ml_kappa(dim=fan_in, eps=self.model.eps)
            
            if 'fc' in trainer_name:
                std_init = math.sqrt((2 / fan_in) * self.model.ratio)
            if 'conv' in trainer_name:
                std_init = math.sqrt((2 / fan_out) * self.model.ratio)
            
            saver_weight_strength = (std_init / saver_rad_sigma)

            if len(saver_dir_loc.shape) == 4:
                out_features, in_features, _, _ = saver_dir_loc.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features,1,1)
                prev_strength = prev_weight_strength.permute(1,0,2,3).expand(out_features,in_features,1,1)
            
            else:
                out_features, in_features = saver_dir_loc.shape
                curr_strength = saver_weight_strength.expand(out_features,in_features)
                if len(prev_weight_strength.shape) == 4:
                    feature_size = in_features // (prev_weight_strength.shape[0])
                    prev_weight_strength = prev_weight_strength.reshape(prev_weight_strength.shape[0],-1)
                    prev_weight_strength = prev_weight_strength.expand(prev_weight_strength.shape[0], feature_size)
                    prev_weight_strength = prev_weight_strength.reshape(-1,1)
                prev_strength = prev_weight_strength.permute(1,0).expand(out_features,in_features)
            
            L2_strength = torch.max(curr_strength, prev_strength) #(4)
            #L2_strength = (1.0 / saver_weight_sigma) #(3a)
            bias_strength = torch.squeeze(saver_weight_strength)
            rad_mu_strength = torch.squeeze(saver_weight_strength)
       
            L1_sigma = saver_rad_sigma
            bias_sigma = torch.squeeze(saver_rad_sigma)
            
            prev_weight_strength = saver_weight_strength
            
            dir_loc_reg = (L2_strength * (trainer_dir_loc - saver_dir_loc)).norm(2)**2
            mu_bias_reg = (bias_strength * (trainer_bias-saver_bias)).norm(2)**2
            rad_mu_reg = (rad_mu_strength * (trainer_rad_mu-saver_rad_mu)).norm(2)**2
            # (5)
            L1_rad_mu_reg = (torch.div(saver_rad_mu**2,L1_sigma**2)*(trainer_rad_mu - saver_rad_mu)).norm(1)
            L1_mu_bias_reg = (torch.div(saver_bias**2,bias_sigma**2)*(trainer_bias - saver_bias)).norm(1)
            
            L1_rad_mu_reg = L1_rad_mu_reg * (std_init ** 2)
            L1_mu_bias_reg = L1_mu_bias_reg * (std_init ** 2)
            #
            rad_sigma = (trainer_rad_sigma**2 / saver_rad_sigma**2)
            
            normal_rad_sigma = trainer_rad_sigma**2
            
            rad_sigma_reg_sum = rad_sigma_reg_sum + (rad_sigma - torch.log(rad_sigma)).sum() # (3b) 
            # rad_sigma_normal_reg_sum = rad_sigma_normal_reg_sum + (normal_rad_sigma - torch.log(normal_rad_sigma)).sum() #(6)
            
            # dir_loc_reg_sum = dir_loc_reg_sum + dir_loc_reg
            mu_bias_reg_sum = mu_bias_reg_sum + mu_bias_reg
            rad_mu_reg_sum = rad_mu_reg_sum + rad_mu_reg
            L1_rad_mu_reg_sum = L1_rad_mu_reg_sum + L1_rad_mu_reg
            L1_mu_bias_reg_sum = L1_mu_bias_reg_sum + L1_mu_bias_reg
            
        # elbo loss
        loss = loss / mini_batch_size
        # L2 loss
        loss = loss + alpha * (mu_bias_reg_sum + rad_mu_reg_sum) / (2 * mini_batch_size)

        # loss = loss + self.saved * dir_loc_reg_sum / (mini_batch_size)
        # L1 loss
        loss = loss + self.saved * (L1_rad_mu_reg_sum + L1_mu_bias_reg_sum) / (mini_batch_size)
        # sigma regularization
        loss = loss + alpha * (rad_sigma_reg_sum) / (mini_batch_size) 

        q_dist = PowerSpherical(trainer_dir_loc, trainer_dir_concentration)
        p_dist = PowerSpherical(saver_dir_loc, saver_dir_concentration)
        kld_dir = KL_Powerspherical(q_dist, p_dist)

        # reg_strength = L2_strength if self.saved else 1
        # kld_dir = KL_vMF_kappa_full(trainer_dir_loc, trainer_dir_concentration, saver_dir_loc, saver_dir_concentration, 1)
        loss = loss + alpha * kld_dir.sum() / (mini_batch_size)
            
        return loss

