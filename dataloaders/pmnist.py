import os, sys
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle

########################################################################################################################

def get(seed=0, tasknum = 10):

    filename = 'pMNIST_{}_task.data'.format(tasknum)
    if os.path.isfile('./data/%s'%filename):
        data = torch.load('./data/%s'%filename)
        return data
        
    np.random.seed(seed)
    data = {}
    size = [1, 28, 28]
    # Pre-load
    # MNIST
    mean = torch.Tensor([0.1307])
    std = torch.Tensor([0.3081])
    dat = {}
    dat['train'] = datasets.MNIST('./data/', train=True, download=True)
    dat['test'] = datasets.MNIST('./data/', train=False, download=True)
    
    for i in range(tasknum):
        print(i, end=',')
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'pMNIST_task_{}'.format(i)
        permutation = np.random.permutation(28*28)
        for s in ['train', 'test']:
            arr = dat[s].data.view(dat[s].data.shape[0],-1).float()
            label = torch.LongTensor(dat[s].targets)

            arr = (arr/255 - mean) / std
            data[i][s]={}
            data[i][s]['x'] = arr[:,permutation].view(-1, size[0], size[1], size[2])
            data[i][s]['y'] = label

        r = np.random.permutation(data[i]['train']['x'].size(0))
        data[i]['valid'] = {}
        data[i]['valid']['x'] = data[i]['train']['x'][r[50000:]].clone()
        data[i]['valid']['y'] = data[i]['train']['y'][r[50000:]].clone()
        data[i]['train']['x'] = data[i]['train']['x'][r[:50000]].clone()
        data[i]['train']['y'] = data[i]['train']['y'][r[:50000]].clone()

    print('done')
    torch.save(data, './data/%s'%filename)

    return data

########################################################################################################################