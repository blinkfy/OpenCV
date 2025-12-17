import cv2
import numpy as np
A = cv2.imread('apple.jpg')
B = cv2.imread('orange.jpg')
B=cv2.resize(B,(A.shape[1],A.shape[0]))
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpA[i])
    # ensure GE has same size as gpA[i-1] (pyrUp may produce size differences for odd dims)
    if GE.shape[:2] != gpA[i-1].shape[:2]:
        GE = cv2.resize(GE, (gpA[i-1].shape[1], gpA[i-1].shape[0]))
    L = cv2.subtract(gpA[i-1], GE)
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5,0,-1):
    GE = cv2.pyrUp(gpB[i])
    if GE.shape[:2] != gpB[i-1].shape[:2]:
        GE = cv2.resize(GE, (gpB[i-1].shape[1], gpB[i-1].shape[0]))
    L = cv2.subtract(gpB[i-1], GE)
    lpB.append(L)
# Now add left and right halves of images in each level
#numpy.hstack(tup)
#Take a sequence of arrays and stack them horizontally to make a single array.
LS = []
for la,lb in zip(lpA,lpB):
    rows,cols,dpt = la.shape
    # use integer half-split, ensure shapes match
    half = cols // 2
    left = la[:, :half]
    right = lb[:, half:cols]
    # if left/right heights differ, resize to match
    if left.shape[0] != right.shape[0]:
        h = min(left.shape[0], right.shape[0])
        left = left[:h]
        right = right[:h]
    ls = np.hstack((left, right))
    LS.append(ls)
# now reconstruct
ls_ = LS[0]
for i in range(1, len(LS)):
    ls_ = cv2.pyrUp(ls_)
    # ensure ls_ and LS[i] have same size before adding
    if ls_.shape[:2] != LS[i].shape[:2]:
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
    ls_ = cv2.add(ls_, LS[i])
cv2.imshow('Pyramid_blending',ls_)

Agray=cv2.cvtColor(A,cv2.COLOR_BGR2GRAY)
Adft = cv2.dft(np.float32(Agray),flags = cv2.DFT_COMPLEX_OUTPUT) 	#傅里叶变换
Adft_shift = np.fft.fftshift(Adft)								#将低频值移到中心
yuv=cv2.cvtColor(B,cv2.COLOR_BGR2YUV)
Bgray,u,v=cv2.split(yuv)
Bdft = cv2.dft(np.float32(Bgray),flags = cv2.DFT_COMPLEX_OUTPUT)
Bdft_shift = np.fft.fftshift(Bdft)
mask = np.zeros(Agray.shape, dtype=np.uint8)
cv2.circle(mask,(Agray.shape[1]//2,Agray.shape[0]//2),3,1,-1)
Adft_shift_masked = Adft_shift * mask[:,:,np.newaxis]
Bdft_shift_masked = Bdft_shift * (1 - mask)[:,:,np.newaxis]

aldft_ishift=np.fft.ifftshift(Adft_shift_masked)
bldft_ishift=np.fft.ifftshift(Bdft_shift_masked)
alimg=cv2.idft(aldft_ishift)
blimg=cv2.idft(bldft_ishift)
iaDft=cv2.magnitude(alimg[:,:,0],alimg[:,:,1])
ibDft=cv2.magnitude(blimg[:,:,0],blimg[:,:,1])
ia=np.uint8(iaDft/iaDft.max() * 175)
ib=np.uint8(ibDft/ibDft.max() * 175)
mask=cv2.imread('apple-mask.png',cv2.IMREAD_GRAYSCALE)
y_sum = cv2.add(ia, ib)
mask_f = (mask.astype(np.float32)) / 255.0
blend_float = y_sum.astype(np.float32) * (1.0 - mask_f) + Agray.astype(np.float32) * mask_f
y_uint8 = np.clip(blend_float, 0, 255).astype(np.uint8)
bgr_yuv = cv2.merge([y_uint8, u, v])
bimg = cv2.cvtColor(bgr_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Fourier_blending',bimg)

A=cv2.blur(A,(15,15))
B2=cv2.blur(B,(55,55))
B=cv2.subtract(B,B2)*1.5
cv2.imshow('Gaussian_blending',cv2.add(A, B.astype(np.uint8)))
cv2.imwrite('blending.png',np.hstack([ls_,bimg, cv2.add(A, B.astype(np.uint8))]))
cv2.waitKey(0)