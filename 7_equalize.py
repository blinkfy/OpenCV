#直方图、均衡化、直方图反投影、傅里叶变换
import cv2
import numpy as np
import matplotlib.pyplot as plt
img=cv2.imread('read.jpg')

imgTemp=img[100:400,100:400]
result=cv2.matchTemplate(img,imgTemp,cv2.TM_CCOEFF_NORMED)#模板匹配
minval,maxval,minloc,maxloc=cv2.minMaxLoc(result)
loc = np.where( result >= 0.8)
for pt in zip(*loc[::-1]):
    print(pt)
    cv2.rectangle(img,pt,(pt[0]+imgTemp.shape[1],pt[1]+imgTemp.shape[0]),(0,0,255),1)
cv2.imshow('imgTemp',imgTemp)


#直方图
#plt.hist(img.ravel(),256,[0,256])#对于灰度图
color = ('b','g','r')
# 对一个列表或数组既要遍历索引又要遍历元素时使用内置enumerate函数会有更加直接,优美的做法enumerate 会将数组或列表组成一个索引序列。使我们再获取索引和索引内容的时候更加方便
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
plt.xlim([0,256]) 


#均衡化
equ=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
yuv_y=cv2.equalizeHist(equ[:,:,0])
equ[:,:,0]=yuv_y
equ=cv2.cvtColor(equ,cv2.COLOR_YUV2BGR)


#CLAHE
yuv=cv2.cvtColor(img,cv2.COLOR_BGR2YUV)
clahe=cv2.createCLAHE(2,(8,8))
clb=clahe.apply(yuv[:,:,0])
yuv[:,:,0]=clb
cla=cv2.cvtColor(yuv,cv2.COLOR_YUV2BGR)
cv2.namedWindow('origin-equalize-CLAHE',cv2.WINDOW_NORMAL)
cv2.resizeWindow('origin-equalize-CLAHE',1920,480)
img3=np.hstack([img,equ,cla])
cv2.imshow('origin-equalize-CLAHE',img3)


#直方图反投影 numpy实现
#roi is the object or region of object we need to find
roi = cv2.imread('ppt.jpg')
assert roi is not None, "file could not be read, check with os.path.exists()"
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
 
#target is the image we search in
target = cv2.imread('read.jpg')
target=cv2.resize(target,(640,480))
assert target is not None, "file could not be read, check with os.path.exists()"
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
# Find the histograms using calcHist. Can be done with np.histogram2d also
M = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
I = cv2.calcHist([hsvt],[0, 1], None, [180, 256], [0, 180, 0, 256] )
h,s,v = cv2.split(hsvt)
cv2.normalize(M, M, 0, 1, cv2.NORM_MINMAX)
B = M[h.ravel(),s.ravel()]
B = np.minimum(B,1)
B = B.reshape(hsvt.shape[:2])
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(B,-1,disc,B)
B = np.uint8(B)
B=cv2.normalize(B,B,0,255,cv2.NORM_MINMAX)
#B=(B-B.min())*255/(B.max()-B.min())
ret,thresh = cv2.threshold(B,40,255,0)
cv2.imshow('backprojection-Numpy',thresh)

#OpenCV实现
roi = cv2.imread('ppt.jpg')
assert roi is not None, "file could not be read, check with os.path.exists()"
hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
 
target = cv2.imread('read.jpg')
target=cv2.resize(target,(640,480))
assert target is not None, "file could not be read, check with os.path.exists()"
hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
 
# calculating object histogram
roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256] )
 
# normalize histogram and apply backprojection
cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
 
# Now convolute with circular disc
disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
cv2.filter2D(dst,-1,disc,dst)
 
# threshold and binary AND
ret,thresh = cv2.threshold(dst,100,255,0)
thresh = cv2.merge((thresh,thresh,thresh))
res = cv2.bitwise_and(target,thresh)
 
res = np.hstack((target,res))
cv2.imshow('backprojection-Opencv',res)

#傅里叶
imgg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
dft = cv2.dft(np.float32(imgg),flags = cv2.DFT_COMPLEX_OUTPUT) 	#傅里叶变换（opencv）
dft_shift = np.fft.fftshift(dft)								#将低频值移到中心
magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))#opencv方法，需与变换对应
plt.subplot(121),plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray') 
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([]) 
plt.show()