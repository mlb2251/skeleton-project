import sys,os
import mlb
import plot,test,train,fix
from util import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms.functional as FF
import einops as eo
from tqdm import tqdm


class Config:
    seed = 1000
    num_train = None
    num_test = None
    valid_frac = .2
    load_train = True
    load_test = True
    load_valid = True
    train_workers = 2
    test_workers = 0
    valid_workers = 0
    device = 6
cfg = Config()

torch.manual_seed(cfg.seed)
# warning: these may slow down your model
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(cfg.seed)
random.seed(cfg.seed)





class MNIST(Dataset):
    validdata, testdata, traindata = None, None, None
    _loaded = False
    @classmethod
    def load(cls, cfg):
        """
        populates Dataset.validdata, Dataset.traindata and Dataset.testdata
            - cfg.load_test = False will suppress loading the test data etc
            - note that loads happen into the parent class itself so they'll be shared by all instances
                which is mainly done so the valid loader and train loader don't need to load the same file
                twice if they use the same data
        """
        if cls._loaded:
            return
        validdata, testdata, traindata = None, None, None
        # download the data
        if cfg.load_train:
            ## load the training data
            _traindata = torchvision.datasets.MNIST(root='./data', train=True, download=True)
            if cfg.num_train is not None:
                _traindata = _traindata[:cfg.num_train]
            num_valid = int(len(_traindata)*cfg.valid_frac)
            num_train = len(_traindata) - num_valid
            validdata, traindata =  random_split(_traindata, [num_valid,num_train], generator=torch.Generator().manual_seed(cfg.seed))
        if cfg.load_test:
            ## load the testing data
            testdata = torchvision.datasets.MNIST(root='./data', train=False, download=True)
            if cfg.num_test is not None:
                testdata = testset[:cfg.num_test]
        if cfg.load_valid:
            ## load the valid data
            pass # for this dataset thats already handled by load_train
        cls.validdata = validdata
        cls.testdata = testdata
        cls.traindata = traindata
        cls._loaded = True
        print("Loaded data")

    def __init__(self, mode, cfg):
        self.load(cfg)
        self.data = {
            'train':self.traindata,
            'test':self.testdata,
            'valid':self.validdata,
        }[mode]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        img, label = self.data[i]
        img = FF.pil_to_tensor(img).float()
        img = FF.normalize(img, [.5], [.5]) # 1 channel so singleton lists for mean, std
        label = torch.as_tensor(label).long()
        return img, label




trainset = MNIST('train',cfg)
testset = MNIST('test',cfg)
validset = MNIST('valid',cfg)


trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=cfg.train_workers)
validloader = DataLoader(validset, batch_size=64, shuffle=True, num_workers=cfg.valid_workers)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=cfg.test_workers)



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = eo.rearrange(x, 'b h w c -> b (h w c)') # flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


with torch.cuda.device(cfg.device):

    net = Net()
    net.cuda()


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, 0), total=len(trainloader.dataset)/trainloader.batch_size, ncols=80):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                tqdm.write('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

