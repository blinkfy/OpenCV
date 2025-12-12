#霍夫变换（直线、随机取点以加快速度、圆形）
import cv2
import numpy as np
 
img = cv2.imread('test.png')
img=cv2.resize(img,(480,480))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
edges = cv2.Canny(gray,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
cv2.imshow('canny',edges)
for rho,theta in [x[0] for x in lines]: 
    a = np.cos(theta)
    b = np.sin(theta) 
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a)) 
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1,16)
cv2.imshow('houghlines3',img)

#Probabilistic Hough Transform
minLineLength = 100
maxLineGap = 15
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
img = cv2.imread('test.png')
img=cv2.resize(img,(480,480))
for x1,y1,x2,y2 in [x[0] for x in lines]: 
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),3,16)

#圆形霍夫
gary = cv2.medianBlur(gray,5)
circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=50,maxRadius=550)
if not circles is None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
 
cv2.imshow('Probabilistic Hough+circles',img) 

cv2.waitKey(0) 
cv2.destroyAllWindows()
