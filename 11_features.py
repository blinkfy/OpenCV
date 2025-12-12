# 角点检测 、SIFT 、 FAST 、 BRIEF 、 ORB
import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('jiangyvle.jpg')
imgsg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgg = np.float32(imgsg)
dst = cv2.cornerHarris(imgg,2,3,0.04)
print(dst)
# 标记出角点
for i,line in enumerate(dst > 0.02 * dst.max()):
    for j,row in enumerate(line):
        if row:
            cv2.circle(img,(j,i),2,(255,0,0),-1)
            
#亚像素级别
dst = cv2.dilate(dst,None)
ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0) 
dst = np.uint8(dst)
# 查找重心
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
# 定义停止和细化角点的标准
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# 返回值由角点坐标组成的一个数组（而非图像）
corners = cv2.cornerSubPix(imgg,np.float32(centroids),(5,5),(-1,-1),criteria)
# Now draw them
res = np.hstack((centroids,corners))
#np.int0 可以用来省略小数点后面的数字（非四舍五入）。 res = np.int0(res) img[res[:,1],res[:,0]]=[0,0,255]
img[np.int0(res[:, 3]), np.int0(res[:, 2])] = [0,255,0]

corners = cv2.goodFeaturesToTrack(imgsg,25,0.01,10)
# 返回的结果是  [[ 311.,  250.]] 两层括号的数组。
corners = np.int0(corners)
for i in corners: 
    x,y = i.ravel()
    cv2.circle(img,(x,y),2,(0,0,255),-1) 

cv2.imshow('corner',img)

plt.show()
plt.show()
cv2.waitKey(0)


#--------------------------------SIFT---------------------------------------------
img = cv2.imread('jiangyvle.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建 SIFT 算法
# nfeatures：特征层数
# nOctaveLayers：高斯金字塔 octave 组数
# contrastThreshold：极值点阈值
# edgeThreshold：边缘效应阈值
# SIFT_create([, nfeatures[, nOctaveLayers[, contrastThreshold[, edgeThreshold[, sigma]]]]]) -> retval
sift= cv2.SIFT.create()
# 查找关键点位置
kp = sift.detect(imgGray,None)
# 计算特征
# keypoints：所有关键点
# descriptors：关键点的描述符
# compute(img,KeyPoints:tuple) -> KeyPoints:tuple, descriptors:np.ndarray
kp,des = sift.compute(imgGray,kp)
# 将上面两步骤合并为一个函数
kp,des = sift.detectAndCompute(imgGray,None)
# 绘制关键点
# drawKeypoints(image, keypoints, outImage[, color[, flags]]) -> outImage
cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('img',img)
# 得到的是 KeyPoint 类的一个元组
kps = sift.detect(img,None)
kp = kps[0]
# 坐标位置
# 关键点的角度
# 关键点的幅值
print(kp,kp.pt,kp.angle,kp.response,des)
cv2.waitKey(0)
#-----------------------FAST-------------------------------------------
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('jiangyvle.jpg', 0)
# Initiate FAST object with default values
fast = cv2.FastFeatureDetector.create()
# find and draw the keypoints
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp,None, color=(255, 0, 0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv2.imshow('fast_true', img2)
# Disable nonmaxSuppression 
fast.setNonmaxSuppression(0)
kp = fast.detect(img,None)
print("Total Keypoints without nonmaxSuppression: ", len(kp))
img3 = cv2.drawKeypoints(img, kp,None, color=(255, 0, 0))
cv2.imshow('fast_false', img3)
cv2.waitKey(0)
#-------------------------------BRIEF-----------------------------------

import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('jiangyvle.jpg', cv2.IMREAD_GRAYSCALE)
 
# Initiate STAR detector
star = cv2.xfeatures2d.StarDetector.create()
 
# Initiate BRIEF extractor
brief = cv2.xfeatures2d.BriefDescriptorExtractor.create()
 
# find the keypoints with STAR
kp = star.detect(img,None)
 
# compute the descriptors with BRIEF
kp, des = brief.compute(img, kp)

print( brief.descriptorSize() )
print( des.shape )
img3 = cv2.drawKeypoints(img, kp,None, color=(255, 0, 0))
cv2.imshow('BRIEF', img3)
cv2.waitKey(0)
#--------------------------------ORB------------------------------------
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img = cv2.imread('jiangyvle.jpg', cv2.IMREAD_GRAYSCALE)
 
# Initiate ORB detector  # 初始化 ORB 检测器
orb = cv2.ORB.create()
 
# 使用 ORB 找到关键点
kp ,des= orb.detectAndCompute(img,None)
 
 
# 仅绘制关键点位置，不绘制大小和方向
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
cv2.imshow('ORB',img2)
cv2.waitKey(0)
