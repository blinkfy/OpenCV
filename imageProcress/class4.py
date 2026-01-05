import cv2
import numpy as np
def dft_process(img):
    yuv=cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    u_channel = yuv[:, :, 1]
    v_channel = yuv[:, :, 2]
    gray = yuv[:, :, 0]
    # DFT
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    mag, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])

    h, w = gray.shape
    Yf, Xf = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dist2 = (Xf - cx)**2 + (Yf - cy)**2
    sigma = min(h, w) * 0.1
    freq_mask = np.exp(-dist2 / (2.0 * sigma * sigma)).astype(np.float32)
    mag *= 1.5-freq_mask

    dft_shift[:, :, 0], dft_shift[:, :, 1] = cv2.polarToCart(mag, phase)
    f_ishift = np.fft.ifftshift(dft_shift)
    idft_real = cv2.idft(f_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)

    gray_back = cv2.normalize(idft_real, None, 0, 255, cv2.NORM_MINMAX)
    gray_uint8 = gray_back.astype(np.uint8)
    gray_uint8 = cv2.GaussianBlur(gray_uint8, (3,3), 0)
    yuv_comb = cv2.merge([gray_uint8, u_channel, v_channel])
    return cv2.cvtColor(yuv_comb, cv2.COLOR_YUV2BGR)

def balance_brightness_lab(src, target_mean=188, target_std=None):
	lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
	L, a, b = cv2.split(lab)
	src_mean = L.mean()
	src_std = L.std()
	if target_std is None:
		target_std = src_std/2
	# 匹配到目标均值/方差
	Lm = (L - src_mean) * (target_std / (src_std + 1e-8)) + target_mean
	Lm = np.clip(Lm, 0, 255)
	lab_m = cv2.merge([Lm.astype(np.uint8), a.astype(np.uint8), b.astype(np.uint8)])
	return cv2.cvtColor(lab_m, cv2.COLOR_LAB2BGR)

def tryThreshold(processed):
    cv2.namedWindow('RGB',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('RGB',width=img.shape[1],height=img.shape[0]+60)
    cv2.namedWindow('HSV',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('HSV',width=img.shape[1],height=img.shape[0]+60)
    cv2.createTrackbar('rmax','RGB',0,255,lambda x:None)
    cv2.createTrackbar('rmin','RGB',0,255,lambda x:None)
    cv2.createTrackbar('gmax','RGB',0,255,lambda x:None)
    cv2.createTrackbar('gmin','RGB',0,255,lambda x:None)
    cv2.createTrackbar('bmax','RGB',0,255,lambda x:None)
    cv2.createTrackbar('bmin','RGB',0,255,lambda x:None)
    cv2.createTrackbar('hmin','HSV',0,255,lambda x:None)
    cv2.createTrackbar('hmax','HSV',0,255,lambda x:None)
    cv2.createTrackbar('smin','HSV',0,255,lambda x:None)
    cv2.createTrackbar('smax','HSV',0,255,lambda x:None)
    cv2.createTrackbar('vmin','HSV',0,255,lambda x:None)
    cv2.createTrackbar('vmax','HSV',0,255,lambda x:None)
    cv2.setTrackbarPos('rmax','RGB',255)
    cv2.setTrackbarPos('rmin','RGB',118)
    cv2.setTrackbarPos('gmax','RGB',170)
    cv2.setTrackbarPos('gmin','RGB',128)
    cv2.setTrackbarPos('bmax','RGB',128)
    cv2.setTrackbarPos('hmin','HSV',24)
    cv2.setTrackbarPos('hmax','HSV',55)
    cv2.setTrackbarPos('smin','HSV',47)
    cv2.setTrackbarPos('smax','HSV',255)
    cv2.setTrackbarPos('vmin','HSV',121)
    cv2.setTrackbarPos('vmax','HSV',183)
    while True:
        rmax=cv2.getTrackbarPos('rmax','RGB')
        rmin=cv2.getTrackbarPos('rmin','RGB')
        gmax=cv2.getTrackbarPos('gmax','RGB')
        gmin=cv2.getTrackbarPos('gmin','RGB')
        bmax=cv2.getTrackbarPos('bmax','RGB')
        bmin=cv2.getTrackbarPos('bmin','RGB')
        hmax=cv2.getTrackbarPos('hmax','HSV')
        hmin=cv2.getTrackbarPos('hmin','HSV')
        smax=cv2.getTrackbarPos('smax','HSV')
        smin=cv2.getTrackbarPos('smin','HSV')
        vmax=cv2.getTrackbarPos('vmax','HSV')
        vmin=cv2.getTrackbarPos('vmin','HSV')
        mask=cv2.inRange(processed, np.array([bmin,gmin,rmin]), np.array([bmax,gmax,rmax]))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        mask2=cv2.inRange(cv2.cvtColor(processed,cv2.COLOR_BGR2HSV),np.array([hmin,smin,vmin]),np.array([hmax,smax,vmax]))
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        img1=processed&mask[:,:,np.newaxis]
        img2=processed&mask2[:,:,np.newaxis]
        cv2.imshow('RGB', img1)
        cv2.imshow('HSV', img2)
        if cv2.waitKey(1) & 0xFF==27:
            break

def hist_backproj(processed):
    cv2.namedWindow('water',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('water',width=img.shape[1],height=img.shape[0])
    watermmask=cv2.imread('oil-mask.png', cv2.IMREAD_GRAYSCALE)
    sample=cv2.imread('oil-sample.png')
    hsv=cv2.cvtColor(processed,cv2.COLOR_BGR2HSV)
    if sample is not None:
        sample_hsv=cv2.cvtColor(sample,cv2.COLOR_BGR2HSV)
        roihist=cv2.calcHist([sample_hsv],[0,1],watermmask,[180,256],[0,180,0,256])
    else:
        roihist=cv2.calcHist([hsv],[0,1],watermmask,[180,256],[0,180,0,256])
    cv2.normalize(roihist, roihist,0,255,cv2.NORM_MINMAX)
    dst=cv2.calcBackProject([hsv],[0,1],roihist,[0,180,0,256],1)
    ret,thresh=cv2.threshold(dst,1,255,0)
    thresh=cv2.dilate(thresh,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),iterations=2)
    thresh=cv2.merge((thresh,thresh,thresh))
    return cv2.bitwise_and(processed,thresh)

def k_means(img, K):
    Z=img.reshape((-1, 3))
    Z=np.float32(Z)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    center=np.uint8(center)
    res=center[label.flatten()]
    return res.reshape(img.shape)

if __name__ == '__main__':
    # 示例：对原图做 Lab 均值-方差匹配与 CLAHE 并对比
    img = cv2.imread('stone.png')
    out_lab = balance_brightness_lab(img)
    out_dft = dft_process(out_lab)
    processed=cv2.medianBlur(out_dft,3)
    processed=cv2.bilateralFilter(processed,3,10,10)
    yuv=cv2.cvtColor(processed,cv2.COLOR_BGR2YUV)
    g=yuv[:,:,0]
    # g=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20)).apply(yuv[:,:,0])
    g=cv2.add(g.astype(np.int16),cv2.Sobel(g,cv2.CV_16S,dx=1,dy=1)//4).astype(np.uint8)
    processed=cv2.cvtColor(cv2.merge([g,yuv[:,:,1],yuv[:,:,2]]), cv2.COLOR_YUV2BGR)
    hstacked = np.hstack([img, out_lab, out_dft, processed])
    cv2.putText(hstacked, 'Original', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(hstacked, 'Lab Balanced', (img.shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(hstacked, 'DFT Processed', (img.shape[1]*2 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(hstacked, 'Final Processed', (img.shape[1]*3 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.imshow('preprocess', hstacked)
    # cv2.imshow('histbjproj', hist_backproj(processed))
    # tryThreshold(processed)
    oilmask=cv2.inRange(cv2.cvtColor(processed,cv2.COLOR_BGR2HSV),np.array([24,47,121]),np.array([55,255,183]))
    oilmask=cv2.morphologyEx(oilmask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations=2)
    oilmask=cv2.morphologyEx(oilmask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    oilmask=cv2.morphologyEx(oilmask,cv2.MORPH_DILATE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    cv2.imshow('oil', processed&oilmask[:,:,np.newaxis])
    
    # 将油区填充为周围颜色
    kernel_dilate=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    oilmask_dilated=cv2.dilate(oilmask,kernel_dilate, iterations=3)
    oilmask_d=cv2.dilate(oilmask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3, 3)))
    # 边界 = 膨胀区域 - 原始区域
    boundary=cv2.bitwise_and(oilmask_dilated, cv2.bitwise_not(oilmask))
    # 对每个油区像素，采样其边界邻域的平均色
    for y in range(processed.shape[0]):
        for x in range(processed.shape[1]):
            if oilmask[y,x]>0:
                y1, y2=max(0, y-17),min(processed.shape[0],y+18)
                x1, x2=max(0, x-17),min(processed.shape[1],x+18)
                boundary_local = boundary[y1:y2,x1:x2]
                if boundary_local.sum()>0:
                    processed[y, x]=processed[y1:y2,x1:x2][boundary_local>0].mean(axis=0).astype(np.uint8)
                else:
                    processed[y, x]=processed[oilmask==0].mean(axis=0).astype(np.uint8)
    gray=cv2.cvtColor(processed,cv2.COLOR_BGR2GRAY)
    water_canny=cv2.Canny(gray,57,255,L2gradient=True)
    water_mask=water_canny|cv2.morphologyEx(water_canny,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12)))
    water_mask=cv2.morphologyEx(water_mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    water_mask=cv2.morphologyEx(water_mask,cv2.MORPH_DILATE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    water=processed&water_mask[:,:,np.newaxis]|water_canny[:,:,np.newaxis]&np.full_like(processed,(0,0,255))
    cv2.imshow('water', water)
    # cv2.imshow('kmeans', k_means(processed, K=3))

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    watermask_dilated = cv2.dilate(water_mask, kernel_dilate, iterations=3)
    # 边界 = 膨胀区域 - 原始区域（油周围的邻域）
    boundary = cv2.bitwise_and(watermask_dilated, cv2.bitwise_not(water_mask))
    # 对每个油区像素，采样其边界邻域的平均色
    for y in range(processed.shape[0]):
        for x in range(processed.shape[1]):
            if water_mask[y, x] > 0:
                y1, y2 = max(0, y-17), min(processed.shape[0], y+18)
                x1, x2 = max(0, x-17), min(processed.shape[1], x+18)
                boundary_local = boundary[y1:y2, x1:x2]
                if boundary_local.sum() > 0:
                    processed[y, x] = processed[y1:y2, x1:x2][boundary_local > 0].mean(axis=0).astype(np.uint8)
                else:
                    processed[y, x] = processed[water_mask == 0].mean(axis=0).astype(np.uint8)
    processed=cv2.GaussianBlur(processed,(3,3),0)
    cv2.imshow('filled', processed)
    gray=cv2.cvtColor(processed,cv2.COLOR_BGR2GRAY)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_line = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) 
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_line,iterations=5)
    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_small)
    gap_score = cv2.addWeighted(blackhat, 0.7, grad, 0.6, 0)
    gap_score = cv2.GaussianBlur(gap_score, (3, 3), 0)
    cv2.imshow('gap score', cv2.normalize(gap_score, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8))
    gap_canny=cv2.Canny(gap_score,15,30,L2gradient=True)
    # 细线清理：连接断裂 + 去毛刺
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gap_canny, connectivity=8)
    gap_keep = np.zeros_like(gap_canny)
    min_area = 15
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        # 细长：长宽比大，或者面积相对包围盒很“稀疏”
        aspect = max(w, h) / (min(w, h) + 1e-6)
        fill = area / (w * h + 1e-6)
        if aspect >= 3.0 or fill <= 0.35:
            gap_keep[labels == i] = 255
    gap_canny = gap_keep
    gap_mask=gap_canny|cv2.morphologyEx(gap_canny,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    gap_mask=cv2.morphologyEx(gap_mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    # 细线清理：连接断裂 + 去毛刺
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(gap_mask, connectivity=8)
    gap_keep = np.zeros_like(gap_mask)
    min_area = 15
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        # 细长：长宽比大，或者面积相对包围盒很“稀疏”
        aspect = max(w, h) / (min(w, h) + 1e-6)
        fill = area / (w * h + 1e-6)
        if aspect >= 1.15 or fill <= 0.5:
            gap_keep[labels == i] = 255
    gap_mask = gap_keep
    gap=processed&gap_mask[:,:,np.newaxis]|gap_canny[:,:,np.newaxis]&np.full_like(processed,(0,0,255))
    cv2.imshow('gap', gap)
    cv2.imshow('stone',processed&cv2.bitwise_not(oilmask|water_mask|gap_mask)[:,:,np.newaxis])
    all=np.full_like(processed,(200,200,200))
    all[gap_mask>0]=(220,220,255)
    all[water_mask>0]=(255,200,0)
    all[oilmask>0]=(0,165,255)
    cv2.imshow('all', all)
    cv2.imwrite('segmented_result.png', all)
    oilcontours,_=cv2.findContours(oilmask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    watercontours,_=cv2.findContours(water_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    gapcontours,_=cv2.findContours(gap_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img,oilcontours,-1,(0,165,255),2)
    cv2.drawContours(img,watercontours,-1,(255,200,0),2)
    cv2.drawContours(img,gapcontours,-1,(220,220,255),2)
    cv2.imshow('contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
