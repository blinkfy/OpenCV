import numpy as np
import matplotlib.pyplot as plt
class mycv2:
    COLOR_RGB2LAB=0
    COLOR_BGR2LAB=1
    COLOR_LAB2RGB=2
    COLOR_LAB2BGR=3
    COLOR_RGB2BGR=4
    COLOR_BGR2RGB=5
    IMREAD_GRAYSCALE=0
    IMREAD_COLOR=1
    @staticmethod
    def cvtColor(img,flag):
        if flag==mycv2.COLOR_RGB2LAB or flag==mycv2.COLOR_BGR2LAB:
            #归一化
            img_float=img.astype(np.float32)/255
            # RGB/BGR -> XYZ
            if flag==mycv2.COLOR_BGR2LAB:
                R,G,B=img_float[:,:,2],img_float[:,:,1],img_float[:,:,0]
            else:
                R,G,B=img_float[:,:,0],img_float[:,:,1],img_float[:,:,2]

            # 伽马校正
            def gamma_correction(channel):
                mask=channel>0.04045
                channel=np.where(mask,np.power((channel+0.055)/1.055,2.4),channel/12.92)
                return channel
            R=gamma_correction(R)
            G=gamma_correction(G)
            B=gamma_correction(B)
            # RGB -> XYZ D65标准
            X=R*0.4124564+G*0.3575761+B*0.1804375
            Y=R*0.2126729+G*0.7151522+B*0.0721750
            Z=R*0.0193339+G*0.1191920+B*0.9503041
            # XYZ -> LAB
            def f(t):
                delta=6.0/29.0
                mask=t>delta**3
                return np.where(mask,np.power(t,1.0/3.0),t/(3*delta**2)+4.0/29.0)
            fx=f(X/0.95047)
            fy=f(Y)
            fz=f(Z/1.08883)
            L=116.0*fy-16.0
            a=500.0*(fx-fy)
            b=200.0*(fy-fz)

            result=np.zeros_like(img,dtype=np.float32)
            result[:,:,0]=L*255.0/100.0  # L: [0,100] -> [0,255]
            result[:,:,1]=a+128.0       # a: [-127,127] -> [1,255]
            result[:,:,2]=b+128.0       # b: [-127,127] -> [1,255]
            result=np.clip(result,0,255).astype(np.uint8)
        elif flag==mycv2.COLOR_LAB2RGB or flag==mycv2.COLOR_LAB2BGR:
            # LAB -> XYZ -> RGB/BGR
            # OpenCV LAB格式转回标准LAB
            L=img[:,:,0].astype(np.float32)*100.0/255.0  # [0,255] -> [0,100]
            a=img[:,:,1].astype(np.float32)-128.0           # [0,255] -> [-128,127]
            b=img[:,:,2].astype(np.float32)-128.0           # [0,255] -> [-128,127]

            # LAB to XYZ
            fy=(L+16.0)/116.0
            fx=a/500.0+fy
            fz=fy-b/200.0

            def f_inv(t):
                delta=6.0/29.0
                mask=t>delta
                return np.where(mask,np.power(t,3),3*delta**2*(t-4.0/29.0))
            X=0.95047*f_inv(fx)
            Y=f_inv(fy)
            Z=1.08883*f_inv(fz)

            # XYZ to RGB
            R=X*3.2404542+Y*-1.5371385+Z*-0.4985314
            G=X*-0.9692660+Y*1.8760108+Z*0.0415560
            B=X*0.0556434+Y*-0.2040259+Z*1.0572252

            # 反伽马校正 (linear RGB to sRGB)
            def inv_gamma_correction(channel):
                mask=channel>0.0031308
                # 对于负值或极小值,直接使用线性部分
                result=np.where(mask,1.055*np.power(np.maximum(channel,0),1.0/2.4)-0.055,channel*12.92)
                return result

            R=inv_gamma_correction(R)
            G=inv_gamma_correction(G)
            B=inv_gamma_correction(B)

            # 使用float32数组存储结果,避免uint8溢出
            result=np.zeros_like(img,dtype=np.float32)
            if flag==mycv2.COLOR_LAB2BGR:
                result[:,:,0]=B*255.0
                result[:,:,1]=G*255.0
                result[:,:,2]=R*255.0
            else:  # LAB2RGB
                result[:,:,0]=R*255.0
                result[:,:,1]=G*255.0
                result[:,:,2]=B*255.0

            # 最后统一clip并转为uint8
            result=np.clip(result,0,255).astype(np.uint8)
        elif flag==mycv2.COLOR_RGB2BGR or flag==mycv2.COLOR_BGR2RGB:
            result=np.zeros_like(img)
            result[:,:,0]=img[:,:,2]
            result[:,:,1]=img[:,:,1]
            result[:,:,2]=img[:,:,0]
        return result
    
    def resize(img,size):
        result=np.zeros((size[1],size[0],img.shape[2]),dtype=img.dtype)
        y_scale=img.shape[0]/size[1]
        x_scale=img.shape[1]/size[0]
        for i in range(size[1]):
            for j in range(size[0]):
                src_y=int(i*y_scale)
                src_x=int(j*x_scale)
                result[i,j]=img[src_y,src_x]
        return result
    
    def histogram(img,bins,rangem):
        hist=np.zeros(bins,dtype=int)
        if len(img.shape)==1:
            for pixel in img:
                hist[pixel]+=1
            return hist
        hist_list=[]
        for ch in [img[:,:,i] for i in range(img.shape[2])]:
            hist=mycv2.histogram(ch,bins,rangem)
            hist_list.append(hist)
        return hist_list
    
    def equalizeHist(img):
        if len(img.shape)==2:# 灰度图
            hist=mycv2.histogram(img.flatten(),256,[0,256])
            cdf=hist.cumsum()
            cdf_mapped=np.ma.masked_equal(cdf,0)
            cdf_mapped=(cdf_mapped-cdf_mapped.min())*255/(cdf_mapped.max()-cdf_mapped.min())
            cdf_final=np.ma.filled(cdf_mapped,0).astype(np.uint8)
            result=cdf_final[img]
        else:# 彩色图
            channels =[img[:,:,i] for i in range(img.shape[2])]
            eq_channels=[]
            for ch in channels:
                hist=mycv2.histogram(ch.flatten(),256,[0,256])
                cdf=hist.cumsum()
                cdf_mapped=np.ma.masked_equal(cdf,0)
                cdf_mapped=(cdf_mapped-cdf_mapped.min())*255/(cdf_mapped.max()-cdf_mapped.min())
                cdf_final=np.ma.filled(cdf_mapped,0).astype(np.uint8)
                eq_ch=cdf_final[ch]
                eq_channels.append(eq_ch)
            result=np.stack(eq_channels,axis=2)
        return result
    
    def LUT(img,lut):
        if len(img.shape)==2:
            return lut[img]
        result=np.zeros_like(img)
        for i in range(img.shape[2]):
            result[:,:,i]=lut[img[:,:,i]]
        return result
    
    def CLAHE(img,clipLimit=2.0,tileGridSize=(8,8)):
        if len(img.shape)==2:# 灰度图
            h,w=img.shape
            tile_h=h//tileGridSize[1]
            tile_w=w//tileGridSize[0]
            
            # 为每个tile计算映射函数(LUT)
            lut_map=np.zeros((tileGridSize[1],tileGridSize[0],256),dtype=np.uint8)
            
            for i in range(tileGridSize[1]):
                for j in range(tileGridSize[0]):
                    y0=i*tile_h
                    y1=(i+1)*tile_h if i != tileGridSize[1]-1 else h
                    x0=j*tile_w
                    x1=(j+1)*tile_w if j != tileGridSize[0]-1 else w
                    tile=img[y0:y1,x0:x1]
                    hist=mycv2.histogram(tile.flatten(),256,[0,256])
                    # 裁剪直方图
                    excess=np.sum(np.maximum(hist-clipLimit,0))
                    hist=np.minimum(hist,clipLimit)
                    hist+=excess//256
                    cdf=hist.cumsum()
                    cdf_mapped=np.ma.masked_equal(cdf,0)
                    cdf_mapped=(cdf_mapped-cdf_mapped.min())*255/(cdf_mapped.max()-cdf_mapped.min())
                    cdf_final=np.ma.filled(cdf_mapped,0).astype(np.uint8)
                    lut_map[i,j]=cdf_final
            
            # 使用双线性插值对每个像素进行映射
            result=np.zeros_like(img)
            
            for y in range(h):
                for x in range(w):
                    # 计算当前像素所在tile的浮点坐标
                    tile_y=(y + 0.5) / tile_h - 0.5
                    tile_x=(x + 0.5) / tile_w - 0.5
                    
                    # 限制在有效范围内
                    tile_y=np.clip(tile_y,0,tileGridSize[1] - 1)
                    tile_x=np.clip(tile_x,0,tileGridSize[0] - 1)
                    
                    # 获取四个相邻tile的索引
                    y0=int(np.floor(tile_y))
                    y1=min(y0 + 1,tileGridSize[1] - 1)
                    x0=int(np.floor(tile_x))
                    x1=min(x0 + 1,tileGridSize[0] - 1)
                    
                    # 计算插值权重
                    wy=tile_y - y0
                    wx=tile_x - x0
                    
                    # 获取当前像素值
                    pixel_val=img[y,x]
                    
                    # 从四个相邻tile的LUT中获取映射值
                    val_00=lut_map[y0,x0,pixel_val]
                    val_01=lut_map[y0,x1,pixel_val]
                    val_10=lut_map[y1,x0,pixel_val]
                    val_11=lut_map[y1,x1,pixel_val]
                    
                    # 双线性插值
                    val_0=val_00 * (1 - wx) + val_01 * wx
                    val_1=val_10 * (1 - wx) + val_11 * wx
                    result[y,x]=val_0 * (1 - wy) + val_1 * wy
            
            result=result.astype(np.uint8)
        else:# 彩色图
            channels =[img[:,:,i] for i in range(img.shape[2])]
            clahe_channels=[]
            for ch in channels:
                clahe_ch=mycv2.CLAHE(ch,clipLimit,tileGridSize)
                clahe_channels.append(clahe_ch)
            result=np.stack(clahe_channels,axis=2)
        return result
    
    def imdecode(buf,flags=IMREAD_COLOR):
        #不使用cv2自己实现解码：
        import png
        from io import BytesIO
        byte_io=BytesIO(buf.tobytes())
        reader=png.Reader(file=byte_io)
        width,height,pixels,metadata=reader.read()
        img=np.array(list(pixels))
        if metadata['greyscale']:
            img=img.reshape((height,width))
        else:
            channels=3
            if metadata.get('alpha',False):
                channels=4
            img=img.reshape((height,width,channels))
            if flags==mycv2.IMREAD_GRAYSCALE:
                # 转灰度
                r,g,b=img[:,:,0],img[:,:,1],img[:,:,2]
                gray=(0.114*b+0.587*g+0.299*r).astype(np.uint8)
                return gray
        return img[:,:,:3]
    
    def imread(path):
        f=np.fromfile(path,dtype=np.uint8)
        img=mycv2.imdecode(f)
        return img
import cv2
tree=cv2.imread('./blue.png')[:,:,::-1]
vegetables=cv2.imread('bulb-on.png')[:,:,::-1]
tree=mycv2.resize(tree,vegetables.shape[1::-1])
tree_lab=mycv2.cvtColor(tree,mycv2.COLOR_RGB2LAB)
vegetables_lab=mycv2.cvtColor(vegetables,mycv2.COLOR_RGB2LAB)

def histogram_matching(source,reference):
    source_hist=mycv2.histogram(source.flatten(),256,[0,256])
    reference_hist=mycv2.histogram(reference.flatten(),256,[0,256])
    
    source_cdf=source_hist.cumsum()
    source_cdf=source_cdf/source_cdf[-1]
    reference_cdf=reference_hist.cumsum()
    reference_cdf=reference_cdf/reference_cdf[-1]
    #查找表
    lut=np.zeros(256,dtype=np.uint8)
    #对每个灰度级,找最接近的CDF
    for i in range(256):
        diff=np.abs(reference_cdf-source_cdf[i])
        lut[i]=np.argmin(diff)
    return mycv2.LUT(source,lut)

def adjust_mean_std(source,reference):
    source=source.astype(np.float32)
    source_mean=np.mean(source)
    source_std=np.std(source)
    reference_mean=np.mean(reference)
    reference_std=np.std(reference)
    if source_std>0:
        adjusted=(source-source_mean)*(reference_std/source_std)+reference_mean
    else:
        adjusted=source+(reference_mean-source_mean)
    adjusted=np.clip(adjusted,0,255).astype(np.uint8)
    return adjusted

def reinhard_color_transfer(source_lab,reference_lab):
    """
    Reinhard色彩迁移
    """
    result=source_lab.copy().astype(np.float32)
    
    # 对每个通道进行均值和标准差匹配
    for i in range(3):
        source_mean=np.mean(source_lab[:,:,i])
        source_std=np.std(source_lab[:,:,i])
        ref_mean=np.mean(reference_lab[:,:,i])
        ref_std=np.std(reference_lab[:,:,i])
        if source_std>0:
            result[:,:,i]=(result[:,:,i]-source_mean)*(ref_std/source_std)+ref_mean
        else:
            result[:,:,i]=result[:,:,i]+(ref_mean-source_mean)
    result=np.clip(result,0,255).astype(np.uint8)
    return result

def linear_color_transfer(source,reference):
    """
    使用最小二乘法拟合线性变换
    """
    source_flat=source.flatten().astype(np.float32)
    reference_flat=reference.flatten().astype(np.float32)
    
    #最小二乘法
    n=len(source_flat)
    sum_x=np.sum(source_flat)
    sum_y=np.sum(reference_flat)
    sum_xy=np.sum(source_flat*reference_flat)
    sum_xx=np.sum(source_flat*source_flat)

    a=(n*sum_xy-sum_x*sum_y)/(n*sum_xx-sum_x*sum_x+1e-10)+1
    b=(sum_y-a*sum_x)/n
    result=a*source.astype(np.float32)+b
    result=np.clip(result,0,255).astype(np.uint8)
    return result

def percentile_matching(source,reference,low=5,high=95):
    """
    基于百分位数的颜色匹配
    """
    source_flat=source.flatten()
    reference_flat=reference.flatten()
    #百分位数
    source_min=np.percentile(source_flat,low)
    source_max=np.percentile(source_flat,high)
    ref_min=np.percentile(reference_flat,low)
    ref_max=np.percentile(reference_flat,high)
    #线性映射
    source_float=source.astype(np.float32)
    if source_max-source_min>0:
        result=(source_float-source_min)/(source_max-source_min)*(ref_max-ref_min)+ref_min
    else:
        result=source_float+(ref_min-source_min)
    result=np.clip(result,0,255).astype(np.uint8)
    return result

def adaptive_histogram_matching(source,reference,clip_limit=2.0):
    """
    自适应直方图匹配 - 结合CLAHE的思想
    """
    # 先应用CLAHE增强对比度
    source_enhanced=mycv2.CLAHE(source,clip_limit)
    # 然后进行直方图匹配
    result=histogram_matching(source_enhanced,reference)
    return result

def covariance_color_transfer(source,reference):
    """
    基于协方差矩阵的颜色迁移 (Monge-Kantorovich Linear)
    考虑了通道间的相关性,能更好地保持颜色结构
    """
    source_flat=source.reshape(-1,3).astype(np.float32)
    reference_flat=reference.reshape(-1,3).astype(np.float32)
    
    mu_s=np.mean(source_flat,axis=0)
    mu_t=np.mean(reference_flat,axis=0)
    cov_s=np.cov(source_flat,rowvar=False)
    cov_t=np.cov(reference_flat,rowvar=False)
    
    # SVD分解 Sigma=U*S*U.T
    #变换矩阵 A=U*sqrt(S)*U.T
    us,ss,vhs=np.linalg.svd(cov_s)
    ut,st,vht=np.linalg.svd(cov_t)
    
    #平方根矩阵
    As=us@np.diag(np.sqrt(np.maximum(ss,1e-6)))@us.T
    At=ut@np.diag(np.sqrt(np.maximum(st,1e-6)))@ut.T
    
    # 变换矩阵 M=At*inv(As)
    # 这一步将源分布的形状(协方差)变换为目标分布的形状
    M=At@np.linalg.inv(As)
    
    # 变换: x'=(x - mu_s) * M^T + mu_t
    matched=(source_flat-mu_s)@M.T+mu_t
    matched=np.clip(matched,0,255).astype(np.uint8)
    return matched.reshape(source.shape)

def idt_color_transfer(source,reference,steps=20):
    """
    迭代分布迁移 (IDT - Iterative Distribution Transfer)
    通过多次随机投影和匹配来逼近高维分布匹配
    解决颜色集中时的色差问题效果最好
    """
    source_flat=source.reshape(-1,3).astype(np.float32)
    reference_flat=reference.reshape(-1,3).astype(np.float32)
    
    result=source_flat.copy()
    n_s=len(source_flat)
    n_r=len(reference_flat)
    
    # 预先计算目标数据的排序和坐标,加速插值
    r_sorted=np.sort(reference_flat,axis=0) # 这里的排序只是为了获取范围? 不,IDT需要投影后排序
    
    for _ in range(steps):
        # 生成随机旋转矩阵 (QR分解生成正交矩阵)
        R=np.linalg.qr(np.random.randn(3,3))[0]
        
        # 投影到新坐标系
        s_proj=result @ R
        r_proj=reference_flat @ R
        
        # 对每一列(轴)进行一维分布匹配
        for i in range(3):
            s_col=s_proj[:,i]
            r_col=r_proj[:,i]
            
            # 获取排序索引
            s_indices=np.argsort(s_col)
            
            # 目标数据的排序值
            r_col_sorted=np.sort(r_col)
            
            # 使用线性插值匹配分布
            # 将源数据映射到目标数据的分布上
            # 相当于: s_val -> rank -> t_val
            quantiles=np.linspace(0,1,n_s)
            ref_quantiles=np.linspace(0,1,n_r)
            
            matched_values=np.interp(quantiles,ref_quantiles,r_col_sorted)
            
            # 赋值回去
            s_proj[s_indices,i]=matched_values
            
        # 旋转回原坐标系
        result=s_proj @ R.T
        
    result=np.clip(result,0,255).astype(np.uint8)
    return result.reshape(source.shape)

# 方法1: 仅使用直方图匹配
result_lab_hist=vegetables_lab.copy()
result_lab_hist[:,:,1]=histogram_matching(vegetables_lab[:,:,1],tree_lab[:,:,1])
result_lab_hist[:,:,2]=histogram_matching(vegetables_lab[:,:,2],tree_lab[:,:,2])
result_bgr_hist=mycv2.cvtColor(result_lab_hist,mycv2.COLOR_LAB2RGB)

# 方法2: 仅使用均值和标准差调整
result_lab_stats=vegetables_lab.copy()
result_lab_stats[:,:,1]=adjust_mean_std(vegetables_lab[:,:,1],tree_lab[:,:,1])
result_lab_stats[:,:,2]=adjust_mean_std(vegetables_lab[:,:,2],tree_lab[:,:,2])
result_bgr_stats=mycv2.cvtColor(result_lab_stats,mycv2.COLOR_LAB2RGB)

# 方法4: Reinhard色彩迁移算法
result_lab_reinhard=reinhard_color_transfer(vegetables_lab,tree_lab)
result_bgr_reinhard=mycv2.cvtColor(result_lab_reinhard,mycv2.COLOR_LAB2RGB)

# 方法5: 线性颜色迁移
result_lab_linear=vegetables_lab.copy()
result_lab_linear[:,:,1]=linear_color_transfer(vegetables_lab[:,:,1],tree_lab[:,:,1])
result_lab_linear[:,:,2]=linear_color_transfer(vegetables_lab[:,:,2],tree_lab[:,:,2])
result_bgr_linear=mycv2.cvtColor(result_lab_linear,mycv2.COLOR_LAB2RGB)

# 方法6: 百分位数匹配
result_lab_percentile=vegetables_lab.copy()
result_lab_percentile[:,:,1]=percentile_matching(vegetables_lab[:,:,1],tree_lab[:,:,1])
result_lab_percentile[:,:,2]=percentile_matching(vegetables_lab[:,:,2],tree_lab[:,:,2])
result_bgr_percentile=mycv2.cvtColor(result_lab_percentile,mycv2.COLOR_LAB2RGB)

# 方法7: 自适应直方图匹配
result_lab_adaptive=vegetables_lab.copy()
result_lab_adaptive[:,:,1]=adaptive_histogram_matching(vegetables_lab[:,:,1],tree_lab[:,:,1])
result_lab_adaptive[:,:,2]=adaptive_histogram_matching(vegetables_lab[:,:,2],tree_lab[:,:,2])
result_bgr_adaptive=mycv2.cvtColor(result_lab_adaptive,mycv2.COLOR_LAB2RGB)

# 方法8: 协方差矩阵迁移 (MKL)
result_lab_cov=covariance_color_transfer(vegetables_lab,tree_lab)
result_bgr_cov=mycv2.cvtColor(result_lab_cov,mycv2.COLOR_LAB2RGB)

# 方法9: 迭代分布迁移 (IDT)
result_lab_idt=idt_color_transfer(vegetables_lab,tree_lab,steps=15)
result_bgr_idt=mycv2.cvtColor(result_lab_idt,mycv2.COLOR_LAB2RGB)

# 打印统计信息
print("原始图像:")
print(f"Vegetable A-均值: {np.mean(vegetables_lab[:,:,1]):.2f},标准差: {np.std(vegetables_lab[:,:,1]):.2f}")
print(f"Vegetable B-均值: {np.mean(vegetables_lab[:,:,2]):.2f},标准差: {np.std(vegetables_lab[:,:,2]):.2f}")
print(f"Tree A-均值: {np.mean(tree_lab[:,:,1]):.2f},标准差: {np.std(tree_lab[:,:,1]):.2f}")
print(f"Tree B-均值: {np.mean(tree_lab[:,:,2]):.2f},标准差: {np.std(tree_lab[:,:,2]):.2f}")
print("=" * 60)

methods=[
    ("1-直方图匹配",result_lab_hist),
    ("2-均值标准差",result_lab_stats),
    ("3-Reinhard算法",result_lab_reinhard),
    ("4-线性迁移",result_lab_linear),
    ("5-百分位数匹配",result_lab_percentile),
    ("6-自适应直方图",result_lab_adaptive),
    ("7-协方差迁移",result_lab_cov),
    ("8-IDT迭代迁移",result_lab_idt)
]

for name,result_lab in methods:
    print(f"{name}:")
    print(f"  A-均值: {np.mean(result_lab[:,:,1]):.2f},标准差: {np.std(result_lab[:,:,1]):.2f}")
    print(f"  B-均值: {np.mean(result_lab[:,:,2]):.2f},标准差: {np.std(result_lab[:,:,2]):.2f}")

# 直方图对比
plt.figure(figsize=(15,5))
# A通道直方图
plt.subplot(1,2,1)
plt.hist(vegetables_lab[:,:,1].flatten(),176,[50,226],color='blue',alpha=0.5,label='Vegetable',histtype='step',linewidth=2)
plt.hist(tree_lab[:,:,1].flatten(),176,[50,226],color='green',alpha=1,label='Tree',histtype='step',linewidth=2)
colors=['red','black','brown','pink','gray','cyan','orange','purple']
for i,(name,result_lab) in enumerate(methods):
    if i!=6:continue
    method_num=name.split('-')[0].strip()
    plt.hist(result_lab[:,:,1].flatten(),176,[50,226],color=colors[i],alpha=0.5,label=method_num,histtype='step')
plt.title('A')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.legend()
# B通道直方图
plt.subplot(1,2,2)
plt.hist(vegetables_lab[:,:,2].flatten(),176,[50,226],color='blue',alpha=0.5,label='Vegetable',histtype='step',linewidth=2)
plt.hist(tree_lab[:,:,2].flatten(),176,[50,226],color='green',alpha=1,label='Tree',histtype='step',linewidth=2)
for i,(name,result_lab) in enumerate(methods):
    if i!=6:continue
    method_num=name.split('-')[0].strip()
    plt.hist(result_lab[:,:,2].flatten(),176,[50,226],color=colors[i],alpha=0.5,label=method_num,histtype='step')
plt.title('B')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')
plt.ylim(0,15000)
plt.tight_layout()

# 显示窗口
hv=np.vstack((np.hstack((vegetables,tree,result_bgr_hist,result_bgr_stats,result_bgr_cov)),
              np.hstack((result_bgr_reinhard,result_bgr_linear,result_bgr_percentile,result_bgr_adaptive,result_bgr_idt))))
plt.figure(figsize=(15,6))
plt.imshow(hv)
plt.axis('off')
plt.tight_layout()
plt.show()
