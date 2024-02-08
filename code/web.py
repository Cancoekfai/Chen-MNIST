# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 00:50:01 2024

@author: Zhuohui Chen
"""

import cv2
import torch
from flask import Flask, request, render_template


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files['file']
        file.save('templates/' + file.filename)
        return train_model('templates/' + file.filename)
    return render_template('web.html')

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
        
        pool2 = pool2.contiguous()
        flat = pool2.view(pool2.size(0), -1)
        
        fc1 = self.fc1(flat)
        fc1 = torch.nn.functional.relu(fc1)
        fc2 = self.fc2(fc1)
        fc2 = torch.nn.functional.relu(fc2)
        
        out = self.out(fc2)
        out = torch.softmax(out, dim=1)
        
        return out
    
def train_model(file_path):
    model = LeNet_5()
    model.cuda()
    model.load_state_dict(torch.load('models/Chen_mnist_LeNet_5.h5')) #load model
    
    img = cv2.imread(file_path)
    w0, h0, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #灰度化
    binary = cv2.adaptiveThreshold(~img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 99, -10)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    y1 = y - 50
    h1 = y + h + 50
    x1 = x - 50
    w1 = x + w + 50
    if y - 50 < 0:
        y1 = 0
    if x - 50 < 0:
        x1 = 0
    if y + h + 50 > w0:
        h1 = w0
    if x + w + 50 > h0:
        w1 = h0
    img = img[y1: h1, x1: w1]
    # 裁剪
    cut_img = cv2.resize(img, (28, 28))
    cut_img[cut_img != 255] = 0
    img = torch.unsqueeze(torch.tensor(cut_img), axis=0)
    img = torch.unsqueeze(torch.tensor(img), axis=-1)
    img = img.repeat(1, 1, 1, 3).permute(0, 3, 1, 2)
    img = img / 255
    
    prediction = torch.argmax(model(img.cuda()), axis=1)
    prediction = int(prediction.cpu().numpy())
    return '预测结果为: 数字 ' + str(prediction)

if __name__ == '__main__':
    app.run()