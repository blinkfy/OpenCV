#透视变换和阈值
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
def mouse(event,x,y,flag,userdata):
    print(event,x,y,flag)
img=cv.imread('read.jpg')
cv.namedWindow('img',cv.WINDOW_NORMAL)
cv.resizeWindow('img',640,480)
cv.setMouseCallback('img',mouse,None)
pts1=np.float32([(2370,700),(2438,1768),(3894,1722),(3760,890)])
pts2=np.float32([(0,0),(0,320),(480,320),(480,0)])
M=cv.getPerspectiveTransform(pts1,pts2)
img2=cv.warpPerspective(img,M,(480,320))
for p in pts1:
    cv.circle(img,(int(p[0]),int(p[1])),20,(0,255,0),-1)
pts=np.array(pts1,int)
cv.polylines(img,[pts],1,(0,255,0),12,16)
img3=cv.adaptiveThreshold(cv.cvtColor(img2,cv.COLOR_BGR2GRAY),255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,51,4)
aaa,img4=cv.threshold(cv.cvtColor(img2,cv.COLOR_BGR2GRAY),0,255,cv.THRESH_TRUNC+cv.THRESH_OTSU)
print(aaa)
cv.imshow('img',img)
orgimg=img2
imgb=cv.equalizeHist(img2[:,:,0])#均衡化
imgg=cv.equalizeHist(img2[:,:,1])
imgr=cv.equalizeHist(img2[:,:,2])
img2=cv.merge([imgb,imgg,imgr])
img2=cv.cvtColor(orgimg,cv.COLOR_BGR2HSV)
img2[:,:,2]=cv.equalizeHist(img2[:,:,2])
cv.cvtColor(img2,cv.COLOR_HSV2BGR,img2)

clahe=cv.createCLAHE(2,(8,8))#VLAHE
clb=clahe.apply(orgimg[:,:,0])
clg=clahe.apply(orgimg[:,:,1])
clr=clahe.apply(orgimg[:,:,2])
cla=cv.merge([clb,clg,clr])

cla2=cv.cvtColor(orgimg,cv.COLOR_BGR2HSV)#VLAHE2
cla2[:,:,0]=clahe.apply(cla2[:,:,0])
cla2[:,:,2]=clahe.apply(cla2[:,:,2])
cv.cvtColor(cla2,cv.COLOR_HSV2BGR,cla2)

cv.imshow('org-equalize-clahe',np.hstack([orgimg,img2,cla2]))
cv.imshow('img3',img3)
cv.imshow('img4',img4)
cv.waitKey(0)
cv.destroyAllWindows()
cv.imwrite('ppt.jpg',cla)