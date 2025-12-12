#基础操作
import cv2
import numpy as np
import matplotlib.pyplot as plt
points=np.array([])
a=0
def mouse_callback(event,x,y,flags,userdata:any):
    global points,a
    if event == cv2.EVENT_LBUTTONDOWN:
        a=1
    if a:
        if event == cv2.EVENT_LBUTTONUP:
            a=0
    #if event == cv2.EVENT_LBUTTONDOWN:
        print(event,x,y,flags,userdata)
        points=np.array([*points,(x,y,cv2.getTrackbarPos('b','video'),cv2.getTrackbarPos('g','video'),cv2.getTrackbarPos('r','video'),cv2.getTrackbarPos('bar','video'))])
def onTrackbarChange(value):
    pass
video=cv2.VideoCapture('xm.mp4')
# 设置分辨率
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps=video.get(cv2.CAP_PROP_FPS)
videoForm=cv2.VideoWriter_fourcc(*"mp4v")
videosave=cv2.VideoWriter('out.mp4',videoForm,fps,(1380,820))
cv2.namedWindow('video',cv2.WINDOW_NORMAL)
cv2.resizeWindow('video',width=640,height=520)
cv2.createTrackbar('bar','video',1,255,onTrackbarChange)
cv2.createTrackbar('r','video',255,255,onTrackbarChange)
cv2.createTrackbar('g','video',255,255,onTrackbarChange)
cv2.createTrackbar('b','video',255,255,onTrackbarChange)
cv2.setMouseCallback('video',mouse_callback,"userdata")
img=cv2.imread('computer.png')
while video.isOpened():
    flag,frame=video.read()
    if flag:
        value = cv2.getTrackbarPos('bar','video')
        font=cv2.FONT_HERSHEY_SIMPLEX
        frame=cv2.copyMakeBorder(frame,50,50,50,50,cv2.BORDER_CONSTANT,frame,value=(255,255,255))
        cv2.rotate(frame,cv2.ROTATE_180,frame)
        for i in range(1,len(points)):
            #cv2.circle(frame,points[i][:2],3,points[i][2:].tolist(),-1,16)
            cv2.line(frame,points[i-1][:2],points[i][:2],points[i][2:5].tolist(),points[i][5]+1,16)
        cv2.putText(frame,'hello',(30,500), font, 2,(255,255,255),value,16)
        from PIL import Image,ImageFont,ImageDraw
        canvasBg = Image.fromarray(frame)
        brush = ImageDraw.Draw(canvasBg)
        font = ImageFont.truetype("C:\\users\\29513\\appdata\\local\\microsoft\\windows\\fonts\\云峰飞云体.ttf",size=45)
        brush.text((100,100),"你好",(255,255,255),font)
        frame=np.array(canvasBg)

        cv2.imshow('video',frame)
        videosave.write(frame)
        print(frame.shape,value)
    else: break
    if cv2.waitKey(1)==27:
        break
video.release()
videosave.release()
cv2.destroyAllWindows()
