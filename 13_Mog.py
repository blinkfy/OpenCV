#背景分割、均值漂移和摄像机漂移
import cv2
import numpy as np
cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorKNN(history=10)
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    frame=cv2.morphologyEx(frame,cv2.MORPH_OPEN,np.ones((3,3)))
    contours,hi=cv2.findContours(fgmask,3,2)
    move=0
    for c in contours:
        move+=cv2.contourArea(c)
    cv2.imshow('frame',np.hstack([cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),fgmask])) 
    print(move)
    k = cv2.waitKey(30) & 0xff 
    if k == 27:
        break
cap.release() 
cv2.destroyAllWindows()



cap = cv2.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 250,90,400,125  # 硬编码目标初始位置,(c, r) 是目标区域左上角坐标，w, h 是目标的宽和高。
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((255.,255.,255.))) #过滤掉低亮度和非肤色的颜色，只保留 0-180 之间的 H（色调）通道。
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) #计算 ROI 的 H 通道直方图。
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)#归一化直方图，使得直方图值在 0-255 之间，增强对比度。
# 设置终止标准，进行10次迭代或移动的中心点变化小于 1
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2) 
        cv2.imshow('img2',np.hstack([img2,cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)]))
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break 
    else:
        break
 
cv2.destroyAllWindows() 
cap.release()




cap = cv2.VideoCapture(0)
# take first frame of the video
ret,frame = cap.read()
# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)
# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.))) 
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180]) 
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
while(1):
    ret ,frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # Draw it on image
        pts = cv2.boxPoints(ret) 
        pts = np.int0(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2) 
        cv2.imshow('img2',img2)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break 
        else:
            cv2.imwrite(chr(k)+".jpg",img2)
    else:
        break
 
cv2.destroyAllWindows() 
cap.release()
