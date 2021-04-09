import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time


class Appr(object):

    def __init__(self,model,is_bayesian=False,nepochs=100,sbatch=256,lr=0.001,lr_min=1e-5,lr_factor=3,lr_patience=5,clipgrad=100, optim='Adam'):
        self.model=model.to(device)
        self.name = self.model.name
        self.is_bayesian = is_bayesian

        self.nepochs = nepochs
        self.sbatch = sbatch
        self.lr = lr
        self.lr_min = lr_min
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.optim = optim
        self.smax = 400


        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        if self.optim == 'SGD':
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        if self.optim == 'Adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, xtrain, ytrain, xvalid, yvalid):

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
        
        # Loop epochs
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
                torch.save(self.model.state_dict(),'./trained_model/%s.model'%self.name)
                patience = self.lr_patience
                print(' *', end='')
            
            else:
                patience -= 1
                if patience <= 0:
                    lr /= self.lr_factor
                    print(' lr={:.1e}'.format(lr), end='')
                    if lr < self.lr_min:
                        print()
                        break
                        
                    patience = self.lr_patience
                    self.optimizer = self._get_optimizer(lr)
            print()

        # Restore best
        self.model.load_state_dict(torch.load('./trained_model/%s.model'%self.name))
        

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
            s=(self.smax-1/self.smax)*i/len(r)+1/self.smax
            
            # Forward current model
            if self.is_bayesian:
                outputs = self.model.forward(images, sample=True)
            else:
                outputs = self.model.forward(images, s)

            loss=self.ce(outputs,targets)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.optim == 'SGD':
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
            self.optimizer.step()


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
            outputs = self.model.forward(images, self.smax)
                
            loss=self.ce(outputs,targets)
            values,indices=outputs.max(1)
            hits=(indices==targets).float()#*(values>0.5).float()

            total_loss+=loss.data.cpu().numpy()*len(b)
            total_acc+=hits.sum().data.cpu().numpy()
            total_num+=len(b)

        return total_loss/total_num,total_acc/total_num

