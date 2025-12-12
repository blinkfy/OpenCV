#仿射变换
import cv2
import numpy as np
import matplotlib.pyplot as plt
def onTrackbarChange(x):
    pass
img=cv2.imread('F:/program/python/OpenCV/computer.png')
cv2.namedWindow('img',cv2.WINDOW_NORMAL)
cv2.resizeWindow('img',640,840)
cv2.createTrackbar('x1','img',50,500,onTrackbarChange)
cv2.createTrackbar('y1','img',50,500,onTrackbarChange)
cv2.createTrackbar('x2','img',200,500,onTrackbarChange)
cv2.createTrackbar('y2','img',50,500,onTrackbarChange)
cv2.createTrackbar('x3','img',100,500,onTrackbarChange)
cv2.createTrackbar('y3','img',250,500,onTrackbarChange)
cv2.createTrackbar('x4','img',200,500,onTrackbarChange)
cv2.createTrackbar('y4','img',200,500,onTrackbarChange)
cv2.createTrackbar('rotate','img',0,360,onTrackbarChange)
while 1:
    x1 = cv2.getTrackbarPos('x1','img')
    y1 = cv2.getTrackbarPos('y1','img')
    x2 = cv2.getTrackbarPos('x2','img')
    y2 = cv2.getTrackbarPos('y2','img')
    x3 = cv2.getTrackbarPos('x3','img')
    y3 = cv2.getTrackbarPos('y3','img')
    x4 = cv2.getTrackbarPos('x4','img')
    y4 = cv2.getTrackbarPos('y4','img')
    angle = cv2.getTrackbarPos('rotate','img')
    R=cv2.getRotationMatrix2D((img.shape[0]//2,img.shape[1]//2),angle,1)
    iimg=img.copy()
    pts1=np.float32([[50,50],[200,50],[50,200]])
    pts2=np.float32([[x1,y1],[x2,y2],[x3,y3]])
    M=cv2.getAffineTransform(pts1,pts2)
    M[0][2]=M[0][2]+x4-200
    M[1][2]=M[1][2]+y4-200
    img2=cv2.warpAffine(iimg, M, (img.shape[0]*2,img.shape[1]*2))
    img2=cv2.warpAffine(img2, R, (img.shape[0]*2,img.shape[1]*2))
    for c in pts2.tolist():
        cv2.circle(img2,(int(c[0]),int(c[1])),5,(0,255,0),-1,16)
    for c in pts1.tolist():
        cv2.circle(img2,(int(c[0]),int(c[1])),5,(0,0,255),-1,16)
    cv2.imshow('img',img2)
    if cv2.waitKey(33)==27: break
cv2.destroyAllWindows()