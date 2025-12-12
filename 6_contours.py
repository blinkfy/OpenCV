#图像的轮廓 & 轮廓特征属性
import cv2
import numpy as np
import math
img=cv2.imread('test.png')
img_=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,img_2=cv2.threshold(img_,127,255,cv2.THRESH_BINARY)
print(img_2)
contours,hierarchy=cv2.findContours(img_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img2=cv2.drawContours(img.copy(),contours,-1,(0,0,255),3,16)
for i in contours[0]:   # 即drawContours的contourIdx参数填0
    cv2.circle(img2,i[0],3,(0,255,255),-1)
for i in contours[1]:   # 即drawContours的contourIdx参数填0
    cv2.circle(img2,i[0],3,(0,255,255),-1)

M=cv2.moments(contours[0])
x,y=int(M['m10']/M['m00']),int(M['m01']/M['m00'])
cv2.circle(img2,(x,y),3,(0,0,255),-1)
print('weight:',x,y)

print("square:",cv2.contourArea(contours[0]))
print("length:",cv2.arcLength(contours[0],True))
convex=cv2.convexHull(contours[0])#凸包
for point in convex:
    cv2.circle(img2,point[0],1,(0,255,0),-1)

xy,wh,theta = cv2.minAreaRect(contours[0])#最小矩形点（旋转）
cv2.boxPoints((xy,wh,theta),img2)

center,radius = cv2.minEnclosingCircle(contours[0])#外接圆
radius = int(radius)
cv2.circle(img2,(int(center[0]),int(center[1])),radius,(0,255,0),1,16)

rows,cols = img.shape[:2]#直线拟合
[dx,dy,x,y] = cv2.fitLine(contours[1], cv2.DIST_L2,0,0.01,0.01)
dx,dy,x,x=*dx,*dy,int(*x),int(*y)
lefty = int((-x*dy/dx) + y)
righty = int(((cols-x)*dy/dx)+y)
cv2.line(img2,(cols-1,righty),(0,lefty),(255,255,0),2)

x,y,w,h = cv2.boundingRect(contours[0])#横纵比
print('Aspect Ratio: '+str(float(w)/h))

area = cv2.contourArea(contours[0])#范围(轮廓面积与边界矩形面积的比值)
rect_area = w*h
extent = float(area)/rect_area
print('extent: '+str(extent))

area = cv2.contourArea(contours[0])#固体度
hull = cv2.convexHull(contours[0])
hull_area = cv2.contourArea(hull)
solidity = float(area)/hull_area
print('solidity: '+str(solidity))

area = cv2.contourArea(contours[0])#等效直径(与轮廓面积相等的圆形的直径)
equi_diameter = np.sqrt(4*area/np.pi)
print("equal_diameter: "+str(equi_diameter))

angle=cv2.fitEllipse(contours[1])[2]#方向
print('angle: '+str(angle))
cv2.line(img2,(int(M['m10']/M['m00'])-100,int(M['m01']/M['m00']-100*math.tan(angle))),(int(M['m10']/M['m00'])+100,int(M['m01']/M['m00']+100*math.tan(angle))),(0,255,255),2)

mask = np.zeros(img2.shape[:2],np.uint8)#掩模
cv2.drawContours(mask,[contours[0]],0,255,-1)
pixelpoints = np.transpose(np.nonzero(mask))#获取上面所有像素点坐
#pixelpoints = cv2.findNonZero(mask)同上

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY),mask)#最大值和最小值及它们的位置
print('最大值和最小值及它们的位置:',min_val, max_val, min_loc, max_loc)

mean_val = cv2.mean(img2,mask = mask)#平均颜色及平均灰度
print('平均颜色及平均灰度'+str(mean_val))

leftmost = tuple(contours[0][contours[0][:,:,0].argmin()][0])#极点
rightmost = tuple(contours[0][contours[0][:,:,0].argmax()][0])
topmost = tuple(contours[0][contours[0][:,:,1].argmin()][0])
bottommost = tuple(contours[0][contours[0][:,:,1].argmax()][0])
print('极点'+str(leftmost),str(rightmost),str(topmost),str(bottommost))

cv2.imshow('img',img2)
cv2.waitKey(0)




quit()
video=cv2.VideoCapture(0)
cv2.namedWindow('video',cv2.WINDOW_NORMAL)
cv2.resizeWindow('video',width=640,height=520)
while video.isOpened():
    flag,frame=video.read()
    if flag:
        frame_=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,frame_2=cv2.threshold(frame_,127,255,cv2.THRESH_BINARY)
        contours2,hierarchy=cv2.findContours(frame_2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        frame=cv2.drawContours(frame,contours2,-1,(0,0,255),3,16)
        cv2.imshow("video",frame)

    if cv2.waitKey(1)==27:
        break
video.release()
cv2.destroyAllWindows()