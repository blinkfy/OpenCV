#图像滤波和边缘检测
import cv2
import numpy as np
from matplotlib import pyplot as plt
video=cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def guideFilter(I, p, winsize:tuple, eps=0.01):
    '''引导滤波器
    I:引导图像
    p:输入图像
    winsize:窗口大小
    eps:正则化参数
    '''
    I = I.astype(np.float32)
    p = p.astype(np.float32)
    I=I/255.
    p=p/255.
    # 均值平滑
    mean_I=cv2.blur(I, winsize)
    mean_p= cv2.blur(p, winsize)
    # I*I和I*p的均值平滑
    mean_II =cv2.blur(I*I, winsize)
    mean_Ip = cv2.blur(I*p, winsize)
    #方差
    var_I=mean_II-mean_I*mean_I#方差公式
    cov_Ip =mean_Ip-mean_I*mean_p#协方差
    
    a=cov_Ip /(var_I + eps)
    b = mean_p-a *mean_I
    #对a、b进行均值平滑
    mean_a=cv2.blur(a,winsize)
    mean_b=cv2.blur(b,winsize)
    q =mean_a*I+ mean_b
    q*= 255.
    q=np.clip(q,0,255)
    return q.astype(np.uint8)

while video.isOpened():
    flag,frame=video.read()
    if flag:
        frame=cv2.resize(frame,(frame.shape[1]//2,frame.shape[0]//2))
        kernel = np.full((5,5),1,np.float32)/25
        frame1=cv2.filter2D(frame,-1,kernel)
        frame2=cv2.boxFilter(frame,-1,(5,5))
        frame3=cv2.blur(frame,(5,5))
        frame4=cv2.GaussianBlur(frame,(5,5),0)
        frame5=cv2.medianBlur(frame,5)
        frame6=cv2.bilateralFilter(frame,5,75,75)
        frame71=cv2.Sobel(frame,cv2.CV_16S,1,0)
        frame72=cv2.Sobel(frame,cv2.CV_16S,0,1)
        frame7=cv2.add(cv2.convertScaleAbs(frame71),cv2.convertScaleAbs(frame72))
        frame81=cv2.Scharr(frame,cv2.CV_16S,1,0)
        frame82=cv2.Scharr(frame,cv2.CV_16S,0,1)
        frame8=cv2.add(cv2.convertScaleAbs(frame81),cv2.convertScaleAbs(frame82))
        frame9=cv2.Laplacian(frame,cv2.CV_16S,ksize=5)
        frame9=cv2.convertScaleAbs(frame9)
        frame10=cv2.Canny(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),30,50)
        frame11=guideFilter(frame,frame,(5,5),0.01)
        cv2.imshow('original',frame)
        cv2.imshow('filter2D',frame1)
        cv2.imshow('boxFilter',frame2)
        cv2.imshow('blur',frame3)
        cv2.imshow('GaussianBlur',frame4)
        cv2.imshow('medianBlur',frame5)
        cv2.imshow('bilateralFilter',frame6)
        cv2.imshow('sober',frame7)
        cv2.imshow('scharr',frame8)
        cv2.imshow('laplacian',frame9)
        cv2.imshow('canny',frame10)
        cv2.imshow('guideFilter',frame11)
    if cv2.waitKey(1)==27:
        break
    e2 = cv2.getTickCount()
cv2.destroyAllWindows()