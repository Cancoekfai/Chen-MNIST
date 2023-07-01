# import module
import time
import torch
import sklearn
import torchvision
import numpy as np
from tqdm import tqdm



"""------------------------------------
@@@ Data preparation and processing
------------------------------------"""
# read images
batch_size = 128
image_size = 28
train_dataset = torchvision.datasets.MNIST(root='../MNIST', train=True,
                transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((image_size, image_size))]))
test_dataset = torchvision.datasets.MNIST(root='../MNIST', train=False,
               transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
               torchvision.transforms.Resize((image_size, image_size))]))
train_db = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                       shuffle=False, sampler=range(8000))
val_db = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                     shuffle=False, sampler=range(8000, 9000))
test_db = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                      shuffle=False, sampler=range(1000))



"""----------------
@@@ Build Model
----------------"""
# build model: (PyTorch)
class LeNet_5(torch.nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        # bulid convolution layer1
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                     out_channels=6,
                                     kernel_size=5,
                                     padding=2)
        # build pooling layer1
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
        # bulid convolution layer2
        self.conv2 = torch.nn.Conv2d(in_channels=6,
                                     out_channels=16,
                                     kernel_size=5,
                                     padding=0)
        # build pooling layer1
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        
        # build hidden layer
        self.fc1 = torch.nn.Linear(400, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        
        # build output layer
        self.out = torch.nn.Linear(84, 10)
        
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv1 = torch.nn.functional.relu(conv1)
        pool1 = self.pool1(conv1)
        conv2 = self.conv2(pool1)
        conv2 = torch.nn.functional.relu(conv2)
        pool2 = self.pool2(conv2)
        
        flat = pool2.view(pool2.size(0), -1)
        
        fc1 = self.fc1(flat)
        fc1 = torch.nn.functional.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = torch.nn.functional.relu(fc2)
        
        out = self.out(fc2)
        out = torch.softmax(out, dim=1)
        
        return out
model = LeNet_5()
model.cuda()



"""-------------------------------
@@@ Model train   epochs=20 6s
-------------------------------"""
# setting optimizer
optimizer = torch.optim.Adam(model.parameters())

times = 0
Epoch = 20
for epoch in range(Epoch):
    total = 0
    correct = 0
    with tqdm(total=len(train_db), desc='Epoch {}/{}'.format(epoch+1, Epoch)) as pbar:
        for step, (x, y) in enumerate(train_db):
            t1 = time.time()
            x = x.cuda() #use GPU
            y = y.cuda() #use GPU
            output = model(x)
            # loss function
            CrossEntropyLoss = torch.nn.CrossEntropyLoss()
            loss = CrossEntropyLoss(output, y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # accuracy
            pred = torch.max(output, axis=1)[1]
            total += y.shape[0]
            correct += int((pred == y).sum())
            pbar.set_postfix({'loss': '%.4f'%float(loss),
                              'accuracy': '%.2f'%((correct/total) * 100) + '%'})
            t2 = time.time()
            times += t2 - t1
            
            # verify
            with torch.no_grad():
                if step == len(train_db)-1:
                    Loss_val = []
                    total_val = 0
                    correct_val = 0
                    for x, y in val_db:
                        x = x.cuda()
                        y = y.cuda()
                        output = model(x)
                        # loss function
                        CrossEntropyLoss = torch.nn.CrossEntropyLoss()
                        loss_val = CrossEntropyLoss(output, y.long())
                        Loss_val.append(float(loss_val))
                        # accuracy
                        pred = torch.max(output, axis=1)[1]
                        total_val += y.shape[0]
                        correct_val += int((pred == y).sum())
                    pbar.set_postfix({'loss': '%.4f'%float(loss),
                                      'val_loss': '%.4f'%np.mean(Loss_val),
                                      'accuracy': '%.2f'%((correct/total) * 100) + '%',
                                      'val_accuracy': '%.2f'%((correct_val/total_val) * 100) + '%'})
            pbar.update(1)
print('Time taken: %d seconds'%times)



"""------------------------------
@@@ Model evaluation   88.30%
------------------------------"""
Loss = []
total = 0
correct = 0
with tqdm(total=len(test_db)) as pbar:
    with torch.no_grad():
         for x, y in test_db:
            x = x.cuda() #use GPU
            y = y.cuda() #use GPU
            output = model(x)
            
            # loss function
            CrossEntropyLoss = torch.nn.CrossEntropyLoss()
            loss = CrossEntropyLoss(output, y.long())
            Loss.append(float(loss))
            
            # accuracy
            pred = torch.max(output, axis=1)[1]
            total += y.shape[0]
            correct += int((pred == y).sum())
            
            pbar.set_postfix({'loss': '%.4f'%np.mean(Loss),
                              'accuracy': '%.2f'%((correct/total) * 100) + '%'})
            pbar.update(1)