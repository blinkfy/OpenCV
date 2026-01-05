import cv2
import os
import numpy as np
maskSize=40
faceSize=20
mask=cv2.imread('heart.png',cv2.IMREAD_GRAYSCALE)
mask=cv2.resize(mask,(maskSize,maskSize))
imagesList=os.listdir('faces')
heart=np.zeros((maskSize*faceSize,maskSize*faceSize,3),dtype=np.uint8)
cnt=0
for i,cow in enumerate(mask):
    for j,row in enumerate(cow):
        if row>128:
            img=cv2.imread(f'faces/{imagesList[cnt]}')
            img=cv2.resize(img,(faceSize,faceSize))
            heart[i*faceSize:(i+1)*faceSize,j*faceSize:(j+1)*faceSize]=img
            cnt=cnt+1
heart=cv2.resize(heart,(maskSize*faceSize,maskSize*faceSize*9//10))
cv2.imshow('heart',heart)
cv2.imwrite('heartMade.png',heart)

img2=cv2.imread('img2.png')
img2=cv2.resize(img2,(0,0),fx=1.5,fy=1.5)
print(img2.shape)
faces=np.zeros(img2.shape,dtype=np.uint8)
for i in range(0,img2.shape[0],faceSize):
    for j in range(0,img2.shape[1],faceSize):
        img=cv2.imread(f'faces/{imagesList[cnt]}')
        img=cv2.resize(img,(faceSize,faceSize))
        if img2.shape[0]<i+faceSize:
            if img2.shape[1]<j+faceSize:
                faces[i:img2.shape[0],j:img2.shape[1]]=img[:img2.shape[0]-i,:img2.shape[1]-j]
            else:
                faces[i:img2.shape[0],j:j+faceSize]=img[:img2.shape[0]-i]
        elif img2.shape[1]<j+faceSize:
            faces[i:i+faceSize,j:img2.shape[1]]=img[:,:img2.shape[1]-j]
        else:
            faces[i:i+faceSize,j:j+faceSize]=img
        cnt=cnt+1
img2=cv2.addWeighted(img2,0.8,faces,0.2,0)
cv2.imshow('img2',img2)
cv2.imwrite('img2Made.png',img2)
cv2.waitKey(1)

# 计算每张图片的 RGB 三通道均值
names = []
means_array = []
from concurrent.futures import ThreadPoolExecutor, as_completed

def compute_mean(image):
    path = f'faces/{image}'
    img = cv2.imread(path)
    mean_bgr=img.mean(axis=(0, 1)).astype(np.int16)
    names.append(image)
    means_array.append(mean_bgr)

with ThreadPoolExecutor(max_workers=16) as ex:
    for image in imagesList:
        ex.submit(compute_mean, image)

means_array = np.array(means_array, dtype=np.int16)  # shape (N,3)

faceSize=5
loop=True
change=False
def onTrackbarChange(value):
    global faceSize,change
    faceSize=value
    change=True
bigface=cv2.imread('bigface.jpeg')
bigface=cv2.resize(bigface,(0,0),fx=0.5,fy=0.5)
cv2.namedWindow('bigface',cv2.WINDOW_NORMAL)
cv2.resizeWindow('bigface',width=800*bigface.shape[1]//bigface.shape[0],height=800)
cv2.createTrackbar('faceSize','bigface',5,12,onTrackbarChange)
cv2.setTrackbarMin('faceSize','bigface',1)
while loop:
    change=False
    def inn(i):
        if not loop:
            return
        for j in range(0,bigface.shape[1],faceSize):
            if bigface.shape[0]<i+faceSize:
                if bigface.shape[1]<j+faceSize:
                    imax=bigface.shape[0]
                    jmax=bigface.shape[1]
                else:
                    imax=bigface.shape[0]
                    jmax=j+faceSize
            elif bigface.shape[1]<j+faceSize:
                imax=i+faceSize
                jmax=bigface.shape[1]
            else:
                imax=i+faceSize
                jmax=j+faceSize
            gbr=(bigface[i:imax,j:jmax].mean(axis=(0, 1))).astype(np.int16)
            dists = np.sum(abs(means_array-gbr), axis=1)
            idx = int(np.argmin(dists))
            imgname = names[idx]
            img = cv2.imread(f'faces/{imgname}')
            img = cv2.resize(img, (jmax-j, imax-i)).astype(np.int16)
            bigface[i:imax,j:jmax] = np.clip(img+gbr-means_array[idx], 0, 255).astype(np.uint8)
    with ThreadPoolExecutor(max_workers=16) as ex:
        future=[]
        i=0
        while i<bigface.shape[0]:
            future.append(ex.submit(inn, i))
            i+=faceSize
        for fs in range(0,len(future),4):
            future[fs].result()
            cv2.imshow('bigface', bigface)
            if cv2.waitKey(1) & 0xFF == 27:
                loop=False
                break
        
    def guideFilter(I,p, winsize:tuple, eps=0.01):
        '''引导滤波器
        I:引导图像
        p:输入图像
        winsize:窗口大小
        eps:正则化参数'''
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

    bigface=cv2.bilateralFilter(bigface,9,75,75)
    bigface=guideFilter(cv2.resize(cv2.imread('bigface.jpeg'),(0,0),fx=0.5,fy=0.5),bigface,(7,7),0.01)
    cv2.imshow('filtered', bigface)
    
    while not change and loop:
        if cv2.waitKey(10) & 0xFF == 27:
            loop=False
            break
    cv2.imwrite('bigfaceMade.png',bigface)
    bigface=cv2.imread('bigface.jpeg')
    bigface=cv2.resize(bigface,(0,0),fx=0.5,fy=0.5)
cv2.destroyAllWindows()