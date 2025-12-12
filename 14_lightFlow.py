#光流估计
import cv2
import numpy as np
video=cv2.VideoCapture('xm.mp4')
# 角点检测的参数
feature_params=dict(maxCorners=100,qualityLevel=0.3,minDistance=7,blockSize=7)
# 光流跟踪的参数
lk_params=dict(winSize=(15,15),maxLevel=2,criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03))
# 生成一些随机颜色
color=np.random.randint(0,255,(100,3))
# 读取第一帧并检测角点
flag,old_frame=video.read()
old_frame=cv2.resize(old_frame,(480,300))
old_gary=cv2.cvtColor(old_frame,cv2.COLOR_BGR2GRAY)
p0=cv2.goodFeaturesToTrack(old_gary,mask=None,**feature_params)
mask=np.zeros_like(old_frame)

hsv=np.zeros_like(old_frame)
hsv[...,1]=255
i=0
while(True):
    flag,frame=video.read()
    i+=1
    if i%2==0:continue
    if not flag:break
    
    frame=cv2.resize(frame,(480,300))
    frame_gary=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # 计算光流
    p1,st,err=cv2.calcOpticalFlowPyrLK(old_gary,frame_gary,p0,None,**lk_params)
    flow = cv2.calcOpticalFlowFarneback(old_gary, frame_gary, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 选取跟踪成功的点
    good_new:np.ndarray=p1[st==1]
    good_old=p0[st==1]
    # 绘制跟踪线
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        cv2.line(mask,new.astype(int),old.astype(int),color[i].tolist(),2,16)
        cv2.circle(mask,new.astype(int),5,color[i].tolist(),-1)
        
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    frame=cv2.addWeighted(frame,0.8,rgb,0.4,0)
    cv2.imshow('frame',cv2.add(frame,mask))
    k=cv2.waitKey(1) & 0xff
    if k==27:
        break
    # 更新上一帧
    old_gary=frame_gary.copy()
    #或 p0=good_new.reshape(-1,1,2) #将good_new转换为(n,1,2)的数组
    p0=p1.reshape(-1,1,2)
video.release()
cv2.destroyAllWindows()