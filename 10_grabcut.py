#GrabCut交互式前景提取
import cv2
import numpy as np
img=cv2.imread('jiangyvle.jpg')
mask=np.zeros(img.shape[:2],np.uint8)
bgdModel=np.zeros((1,65),np.float64)
fgdModel=np.zeros((1,65),np.float64)
rect=(130,190,200,450)
cv2.imshow('org',img)
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2=np.where((mask==0)|(mask==2),0,1).astype('uint8')
img2=img.copy()*mask2[:,:,np.newaxis]
cv2.imshow('1',img2)

mask3=cv2.imread('jiangyvle_mask.jpg',0)
mask[mask3==0]=0
mask[mask3==255]=1
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
mask2=np.where((mask==0)|(mask==2),0,1).astype('uint8')
img=img*mask2[:,:,np.newaxis]
#img[mask3==0]=(0,0,0)
#img[mask3==255]=(255,255,255)
cv2.imshow('2',img)

mask[(mask==3)|(mask==1)]=255
cv2.imshow('mask',mask)
bgra=np.zeros((img.shape[0],img.shape[1],4),np.uint8)
bgra[:,:,:3]=img
bgra[:,:,3]=mask
cv2.imwrite('jiangyvle.png',bgra)
cv2.waitKey(0)