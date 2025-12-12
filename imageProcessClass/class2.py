#空域滤波
import cv2
import numpy as np
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

a=cv2.imread('body.png')
a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)
b=cv2.Laplacian(a,cv2.CV_16S,ksize=3)
c=cv2.subtract(a.astype(np.int16),b,dtype=cv2.CV_8U)
b=cv2.convertScaleAbs(b+128).astype(np.uint8)
d1=cv2.Sobel(a,cv2.CV_16S,1,0,scale=0.8)
d2=cv2.Sobel(a,cv2.CV_16S,0,1,scale=0.8)
d=cv2.add(cv2.convertScaleAbs(d1),cv2.convertScaleAbs(d2),dtype=cv2.CV_8U)
e=cv2.bilateralFilter(d,7,75,75)
f=cv2.bitwise_and(e,c)
g=cv2.add(f,a)
h=(g**0.6).astype(np.uint8)
cv2.normalize(h,h,0,255,cv2.NORM_MINMAX)
_,mask=cv2.threshold(h,90,255,cv2.THRESH_BINARY_INV)
mask_rev=cv2.bitwise_not(mask)
i=guideFilter(h,h,(17,17))
# i=cv2.bilateralFilter(h,11,35,35)
i=cv2.add(cv2.bitwise_and(i,mask),cv2.bitwise_and(h,mask_rev))

show=np.hstack([a,b,c,d,e,f,g,h,i])
x=0
for txt in ['a','b','c','d','e','f','g','h','i']:
    cv2.putText(show,txt,(x+10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,255,2)
    cv2.line(show,(x,0),(x,show.shape[0]),255,2)
    x+=a.shape[1]
cv2.imshow('body',cv2.resize(show,(0,0),fx=0.45,fy=0.45))
cv2.waitKey(0)