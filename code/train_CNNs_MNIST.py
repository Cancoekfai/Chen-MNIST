# import module
import time
import torch
import torchvision
import numpy as np
from tqdm import tqdm



"""------------------------------------
@@@ Data preparation and processing
------------------------------------"""
# read images
batch_size = 128
image_size = 32
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
def Model(model_name):
    if model_name == 'VGG-16':
        model = torchvision.models.vgg16(pretrained=True)
        fc_input = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(fc_input, 10)
    if model_name == 'VGG-19':
        model = torchvision.models.vgg19(pretrained=True)
        fc_input = model.classifier[6].in_features
        model.classifier[6] = torch.nn.Linear(fc_input, 10)
        
    if model_name == 'ResNet-50':
        model = torchvision.models.resnet50(pretrained=True)
        fc_input = model.fc.in_features
        model.fc = torch.nn.Linear(fc_input, 10)
    if model_name == 'ResNet-101':
        model = torchvision.models.resnet101(pretrained=True)
        fc_input = model.fc.in_features
        model.fc = torch.nn.Linear(fc_input, 10)
        
    if model_name == 'DenseNet-169':
        model = torchvision.models.densenet169(pretrained=True)
        fc_input = model.classifier.in_features
        model.classifier = torch.nn.Linear(fc_input, 10)
    if model_name == 'DenseNet-201':
        model = torchvision.models.densenet201(pretrained=True)
        fc_input = model.classifier.in_features
        model.classifier = torch.nn.Linear(fc_input, 10)
        
    return model.cuda() #use GPU



"""-------------------------------
@@@ Model train   epochs=20 6s
-------------------------------"""
Times = []
Accuracy = []
models_name = ['VGG-16', 'VGG-19', 'ResNet-50', 'ResNet-101',
               'DenseNet-169', 'DenseNet-201']
for model_name in models_name:
    model = Model(model_name)
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
                x = x.repeat(1, 3, 1, 1).cuda() #use GPU
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
                            x = x.repeat(1, 3, 1, 1).cuda()
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
    Times.append(int(times))
    
    
    
    """------------------------------
    @@@ Model evaluation   98.00%
    ------------------------------"""
    Loss = []
    total = 0
    correct = 0
    with tqdm(total=len(test_db)) as pbar:
        with torch.no_grad():
             for x, y in test_db:
                x = x.repeat(1, 3, 1, 1).cuda() #use GPU
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
                accuracy = '%.2f'%((correct/total) * 100) + '%'
                
                pbar.set_postfix({'loss': '%.4f'%np.mean(Loss), 'accuracy': accuracy})
                pbar.update(1)
    Accuracy.append(accuracy)
print(Times, Accuracy)