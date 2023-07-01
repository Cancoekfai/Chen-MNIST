# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 15:53:28 2021

@author: chenzhuohui
"""

# 导入模块
import os
import re
import cv2
import numpy as np

# 创建文件夹
if not os.path.exists('test_db'):
    os.mkdir('test_db')
# 读取文件夹
pics=os.listdir('figures')
# 读取图片
img=cv2.imread('figures/'+pics[0]) #以第一张图片为例
print(img.shape)

## 识别表格
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #灰度化
print(gray.shape)
# 二值化
binary=cv2.adaptiveThreshold(~gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY,61,-10) #选择blockSize=61（去噪）
print(np.unique(binary))
cv2.imwrite('binary.png',binary)
# 识别横线
rows,cols=binary.shape
scale=25
# 2为线的粗细，若取1，则不能完全去除表格
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(cols//scale,2))
eroded=cv2.erode(binary,kernel,iterations=2)
row=cv2.dilate(eroded,kernel,iterations=20)
cv2.imwrite('row.png',row)
# 识别竖线
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(2,rows//scale)) #2为线的粗细
eroded=cv2.erode(binary,kernel,iterations=2)
col=cv2.dilate(eroded,kernel,iterations=20)
cv2.imwrite('col.png',col)
'''
kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(1,rows//scale))
eroded=cv2.erode(binary,kernel,iterations=1) #若降低，则会将数字也识别成表格
col_error=cv2.dilate(eroded,kernel,iterations=1)
cv2.imwrite('col_error.png',col_error)
'''
# 识别表格
merge=cv2.add(row,col)
cv2.imwrite('table.png',merge)

## 去除表格
img1=~binary
for i in range(merge.shape[0]):
    for j in range(merge.shape[1]):
        if merge[i,j]!=0:
            img1[i-10:i+10,j-10:j+10]=255
cv2.imwrite('del_table.png',img1)
# 中值滤波器——去噪
img2=cv2.medianBlur(img1,7)
cv2.imwrite('medianBlur.png',img2)

## 图片分割（100张）
# 提取轮廓
binary=cv2.adaptiveThreshold(~img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv2.THRESH_BINARY,99,-10)
contours,hierarchy=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
img3=cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB) #转RGB图
cv2.drawContours(img3,contours,-1,(0,0,255),2) #绘制轮廓
cv2.imwrite('drawContours.png',img3)
# 分割
for i in range(len(contours)):
    x,y,w,h=cv2.boundingRect(contours[i])
    if h>10 and w>10:
        imgi=img2[y-20:y+h+20,x-20:x+w+20]
    # 裁剪
    imgi=cv2.resize(imgi,(28,28))
    cv2.imwrite('test_db/'+str(9-i//10)+str(i%10)+'.png',imgi)
    
if os.path.exists('test_db/210.png'):
    os.remove('test_db/210.png')
os.rename('test_db/10.png','test_db/210.png') #改名（缺点）

pics=os.listdir('test_db')
pic_1=list(filter(lambda i:bool(re.findall(r'^1',i)),pics))
for pic in pic_1:
    img1=cv2.imread('test_db/'+pic)
    kernel=np.ones((2,2))
    img2=cv2.dilate(img1,kernel,iterations=1)
    cv2.imwrite('test_db/'+pic,img2)