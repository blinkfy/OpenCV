import cv2
img=cv2.imread('oil-mask.png',cv2.IMREAD_GRAYSCALE)
img=cv2.threshold(img,100,255,cv2.THRESH_BINARY)[1]
# img=cv2.GaussianBlur(img,(3,3),0)
cv2.imwrite('oil-mask.png',img)