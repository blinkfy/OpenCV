SOURCE=['marilyn','einstein']
# SOURCE=['apple','orange']
import cv2
import numpy as np
A = cv2.imread(SOURCE[0]+'.jpg')
A=cv2.resize(A,(512,512))
B = cv2.imread(SOURCE[1]+'.jpg')
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

Agray,u2,v2=cv2.split(cv2.cvtColor(A,cv2.COLOR_BGR2YUV))
Adft = cv2.dft(np.float32(Agray),flags = cv2.DFT_COMPLEX_OUTPUT) 	#傅里叶变换
Adft_shift = np.fft.fftshift(Adft)								#将低频值移到中心
yuv=cv2.cvtColor(B,cv2.COLOR_BGR2YUV)
Bgray,u,v=cv2.split(yuv)
Bdft = cv2.dft(np.float32(Bgray),flags = cv2.DFT_COMPLEX_OUTPUT)
Bdft_shift = np.fft.fftshift(Bdft)

h, w = Agray.shape
Yf, Xf = np.ogrid[:h, :w]
cx, cy = w // 2, h // 2
dist2 = (Xf - cx)**2 + (Yf - cy)**2
sigma = min(h, w) * (0.03 if SOURCE[0] == 'marilyn' else 0.015)
freq_mask = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)

freq_mask_3 = freq_mask[:, :, np.newaxis]
combined_shift = Adft_shift * freq_mask_3 + Bdft_shift * (1.0 - freq_mask_3)

combined = np.fft.ifftshift(combined_shift)
recon = cv2.idft(combined)
recon_real = recon[:, :, 0]
# normalize
recon_real = recon_real - recon_real.min()
if recon_real.max() != 0:
    recon_real = recon_real / recon_real.max()
iy = (recon_real * 255.0).astype(np.uint8)
u = cv2.GaussianBlur((u2//2+u//2), (11, 11), 0)
v = cv2.GaussianBlur((v2//2+v//2), (11, 11), 0)
bgr_yuv = cv2.merge([iy, u, v])
bimg = cv2.cvtColor(bgr_yuv, cv2.COLOR_YUV2BGR)
cv2.imshow('Fourier_blending',bimg)

A=cv2.blur(A/(1.3 if SOURCE[0] == 'marilyn' else 1),(21,21)).astype(np.uint8)
B2=cv2.blur(B,(25,25))
B=cv2.subtract(B,B2)
cv2.imshow('Gaussian_blending',cv2.add(A, B.astype(np.uint8)))
cv2.imwrite('blended.png',np.hstack([ls_,bimg, cv2.add(A, B.astype(np.uint8))]))
cv2.waitKey(0)