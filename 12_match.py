#匹配
import numpy as np
import cv2
from matplotlib import pyplot as plt
 
img1 = cv2.imread(r"IMG_20250206_194205.jpg") # queryImage
img1=cv2.resize(img1,(600,800))
img2 = cv2.imread(r"IMG_20250206_194159.jpg") # trainImage
img2=cv2.resize(img2,(600,800))
#-------------------------------------ORB_BF--------------------------------
# Initiate ORB detector
orb = cv2.ORB.create()
# find the keypoints and descriptors with ORB 
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('ORB-BF',img3)

#------------------------------SIFT_BF-------------------------------------
# Initiate SIFT detector
sift = cv2.SIFT.create()
# find the keypoints and descriptors with SIFT 
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)
# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# Apply ratio test
# 比值测试,首先获取与A 距离最近的点B（最近）和C（次近）,只有当B/C小于阈值时（0.75）才被认为是匹配,因为假设匹配是一一对应的,真正的匹配的理想距离为0
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance: good.append([m])
 
# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good,None, flags=2)
cv2.imshow('SIFT-BF',img3)

#----------------------------FLANN---------------------------------------------

# Initiate SIFT detector
sift = cv2.SIFT.create()
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
 
flann = cv2.FlannBasedMatcher(index_params,search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
 
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
 
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
 
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imshow('FLANN',img3)
#------------------------------单应性变换查找------------------------------------------------

MIN_MATCH_COUNT = 10
sift = cv2.SIFT.create()
 
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
 
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)
 
flann = cv2.FlannBasedMatcher(index_params, search_params)
 
matches = flann.knnMatch(des1,des2,k=2)
 
# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)
if len(good)>MIN_MATCH_COUNT:
    # 获取关键点的坐标
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) 
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # 第三个参数  用于计算单应矩阵的方法。以下是可能的方法:
    # 0 - 使用所有点的常规方法
    # CV_RANSAC - RANSAC-based robust method
    # CV_LMEDS - Least-Median robust method
    # 第四个参数取值范围在  1 到 10, 拒绝一个点对的阈值。原图像的点经过变换后点与目标图像上对应点的误差, 超过误差就认为是  outlier
    # 返回值中  M 为变换矩阵。
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0) 
    matchesMask = mask.ravel().tolist()
    # 获得原图像的高和宽
    h,w = img1.shape[:2]
    # 使用得到的变换矩阵对原图像的四个角进行变换,获得在目标图像上对应的坐标。
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2) 
    dst = cv2.perspectiveTransform(pts,M)
    # 原图像为灰度图
    cv2.polylines(img2,[np.int32(dst)],True,255,5, cv2.LINE_AA)
 
else:
    print ("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
cv2.imshow('Feature Matching + Homography to find Objects',img3)

img1 = cv2.imread(r"IMG_20250206_194205.jpg")
img1=cv2.resize(img1,(600,800))
img2 = cv2.imread(r"IMG_20250206_194159.jpg")
img2=cv2.resize(img2,(600,800))
img1=cv2.warpPerspective(img1,M,(img2.shape[1]*2,img2.shape[0]))
mask = np.all(img1 == [0, 0, 0], axis=-1)

# 找到 img1 中黑色像素的坐标（可以是黑色区域的左上角）
black_pixels = np.where(mask)
# 获取黑色区域的矩形边界（仅作为示例，我们选择最左上角的黑色区域）
top_left_x = np.min(black_pixels[1])  # 左边界
top_left_y = np.min(black_pixels[0])  # 上边界
# 计算放置 img2 的区域
height, width = img2.shape[:2]
# 确保放置区域不会超出 img1 的边界
if top_left_y + height <= img1.shape[0] and top_left_x + width <= img1.shape[1]:
    # 将 img2 拷贝到 img1 中的黑色部分
    print(top_left_x + width)
    img1[top_left_y:top_left_y + height, top_left_x:top_left_x + width] = img2
cv2.imshow('Pieced picture',img1)
cv2.waitKey(0)