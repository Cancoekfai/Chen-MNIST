# import module
import time
import torch
import sklearn
import itertools
import torchvision
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



"""------------------------------------
@@@ Data preparation and processing
------------------------------------"""
# read images
batch_size = 128
image_size = 28
dataset = torchvision.datasets.ImageFolder(root='../data',
          transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
          torchvision.transforms.Resize((image_size, image_size))]))

# split training set, validation set and test set
val_split = 0.1
test_split = 0.1
indices = list(range(len(dataset)))
indices = sklearn.utils.shuffle(indices, random_state=0)
indices_train = indices[int(len(dataset)*(val_split+test_split)):]
indices_val = indices[:int(len(dataset)*val_split)]
indices_test = indices[int(len(dataset)*val_split): int(len(dataset)*(val_split+test_split))]
train_db = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                       shuffle=False, sampler=indices_train)
val_db = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                     shuffle=False, sampler=indices_val)
test_db = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
                                      shuffle=False, sampler=indices_test)



"""----------------
@@@ Build Model
----------------"""
# build model: (PyTorch)
class LeNet_5(torch.nn.Module):
    def __init__(self):
        super(LeNet_5, self).__init__()
        # bulid convolution layer1
        self.conv1 = torch.nn.Conv2d(in_channels=3,
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

LOSS = []
times = 0
Epoch = 20
LOSS_VAL = []
Accuracy = []
Accuracy_val = []
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
        LOSS.append(float(loss))
        LOSS_VAL.append(np.mean(Loss_val))
        Accuracy.append(correct/total)
        Accuracy_val.append(correct_val/total_val)
    train_history = pd.DataFrame({'loss': LOSS, 'val_loss': LOSS_VAL,
                                  'accuracy': Accuracy, 'val_accuracy': Accuracy_val})
print('Time taken: %d seconds'%times)

# plot curve
def show_train_history(type_str, train_type, val_type, name):
    plt.figure(dpi=200)
    plt.plot(train_history[train_type])
    plt.plot(train_history[val_type])
    plt.ylabel(type_str)
    plt.xlabel('Epoch')
    plt.legend(['train', 'val'], loc='best')
    plt.savefig('../article/figures/%s.jpg'%name)
show_train_history('Loss value', 'loss', 'val_loss', 1)    
show_train_history('Accuracy (%)', 'accuracy', 'val_accuracy', 2)



"""------------------------------
@@@ Model evaluation   98.00%
------------------------------"""
Loss = []
total = 0
correct = 0
y_all = np.array([])
prediction = np.array([])
with tqdm(total=len(test_db)) as pbar:
    with torch.no_grad():
         for x, y in test_db:
            x = x.cuda() #use GPU
            y = y.cuda() #use GPU
            y_all = np.append(y_all, y.cpu())
            output = model(x)
            
            # loss function
            CrossEntropyLoss = torch.nn.CrossEntropyLoss()
            loss = CrossEntropyLoss(output, y.long())
            Loss.append(float(loss))
            
            # accuracy
            pred = torch.max(output, axis=1)[1]
            prediction = np.append(prediction, pred.cpu())
            total += y.shape[0]
            correct += int((pred == y).sum())
            
            pbar.set_postfix({'loss': '%.4f'%np.mean(Loss),
                              'accuracy': '%.2f'%((correct/total) * 100) + '%'})
            pbar.update(1)
            
# confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, cmap=plt.cm.Blues):
    plt.figure(dpi=200)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Pred Label')
    
confusion_mtx = confusion_matrix(y_all, prediction)
plot_confusion_matrix(confusion_mtx, range(10))

# classification report
print(classification_report(prediction, y_all))

# save model
torch.save(model.state_dict(), 'models/Chen_mnist_LeNet_5.h5')