import cv2
import numpy as np
a=cv2.imread('body.png')
b=cv2.Laplacian(a,cv2.CV_64F)