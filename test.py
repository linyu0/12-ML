#!/usr/bin/env python
# coding: utf-8
import torchvision
import torch.optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from net import Net, LeNet


testdata = torchvision.datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=False)
testdata_laoder = DataLoader(dataset=testdata, batch_size=128, shuffle=False)

model = LeNet().cuda()
model.load_state_dict(torch.load('./model_adam_30epoch_84_分段衰减两次_b100_预处理.pkl'))

with torch.no_grad():
    test_acc = 0.0
    for i_batch, (data, target) in enumerate(testdata_laoder):
        data, target = data.cuda(), target.cuda().long()
        output = model(data)  
        
        _, pred = output.max(1)
        num_correct = (pred == target).sum().item()
        test_acc += num_correct
    test_acc = test_acc / len(testdata)
    print('Mnist Testing ACC: {:.6f}'.format(test_acc))