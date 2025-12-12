#分水岭算法
import cv2
import numpy as np

img=cv2.imread('coins.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thesh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#获取二值化图像
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thesh,cv2.MORPH_OPEN,kernel)#开运算以去除噪声
opening=cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)#去除噪声
sure_bg=cv2.dilate(opening,kernel,iterations=1)#得到背景
dist=cv2.distanceTransform(opening,1,5)#距离变换，获取每个像素点距离最近的0的距离
ret,sure_fg=cv2.threshold(dist,0.6*dist.max(),255,0)
sure_fg = np.uint8(sure_fg)
unknow=cv2.subtract(sure_bg,sure_fg)

ret,marker1=cv2.connectedComponents(sure_fg)#创建标签,背景为0，其他从1开始编号
markers=marker1+1
markers[unknow==255]=0
markers3=cv2.watershed(img,markers)
cv2.imshow('original原图',img)
cv2.imshow('bagkground背景',cv2.bitwise_not(sure_bg))
cv2.imshow('frontground前景',sure_fg)
cv2.imshow('unknow无法确定的区域',unknow)
img2=cv2.cvtColor(img.copy(),cv2.COLOR_BGR2HSV)
img2[markers3==-1]=(0,255,255)
markers3*=255//markers3.max()
markers3[markers3<0]=0
img2[:,:,0]+=np.uint8(markers3)
cv2.imshow('watershed',cv2.cvtColor(img2,cv2.COLOR_HSV2BGR))
cv2.waitKey(0)