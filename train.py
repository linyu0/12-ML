
# coding: utf-8

import torch.nn as nn
import torchvision
import torch.optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from net import Net, LeNet


traindata = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))]), download=False)
testdata = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))]), download=False)
traindata_loader = DataLoader(dataset=traindata, batch_size=100, shuffle=True)
testdata_laoder = DataLoader(dataset=testdata, batch_size=128, shuffle=False)

model = LeNet().cuda()

epochs = 30

learning_rate = 0.001

momentum = 0.9


criterion = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.96)

# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.00000001)


path = "model_test.pkl"

acc = 0


for epoch in range(epochs):

    train_loss = 0.0
    train_acc = 0.0

    model.train()

    scheduler.step()

    for i_batch, (data, target) in enumerate(traindata_loader):
        optimizer.zero_grad()  
        data, target = data.cuda(), target.cuda().long()
        output = model(data)  
        loss = criterion(output, target)  
        loss.backward()  
        optimizer.step()  
        train_loss += loss.item()  
        
        _, pred = output.max(1)
        
        num_correct = (pred == target).sum().item()
        train_acc += num_correct  
    
    train_loss = train_loss / len(traindata)
    train_acc = train_acc / len(traindata)
    
    print('Epoch:  {}  \tTraining Loss: {:.6f}'.format(epoch + 1, train_loss), 'ACC: {:.6f}'.format(train_acc))

    
    with torch.no_grad():
        test_acc = 0.0
        for i_batch, (data, target) in enumerate(testdata_laoder):
            data, target = data.cuda(), target.cuda().long()
            output = model(data)  
            _, pred = output.max(1)
            num_correct = (pred == target).sum().item()
            test_acc += num_correct
        test_acc = test_acc / len(testdata)
        print('Epoch:  {}  \tTesting ACC: {:.6f}'.format(epoch + 1, test_acc))
        
        if test_acc > acc:
            torch.save(model.state_dict(), path)
            acc = test_acc
            print('save model!!!')

print('model train and test finish!!!')
print('save model!!!')
# torch.save(model.state_dict(), path)
print('finish model saved!!!')
