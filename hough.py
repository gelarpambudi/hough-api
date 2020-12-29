import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math
import os

Alpha_F = 0.1
Alpha_L = 1.0
Alpha_T = 0.3

V_F = 0.5
V_L = 0.2
V_T = 20.0

Num = 10
Beta = 0.1

W = [0.5, 1, 0.5, 1, 0, 1, 0.5, 1, 0.5,]
W = np.array(W, np.float).reshape((3, 3))
M = W

global F
global L
global Y
global T
global Y_AC

def pcnn(input_image):
    src = cv.cvtColor(input_image, cv.COLOR_BGR2GRAY)
    
    dim = src.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float)
    Y_AC = np.zeros( dim, np.float)
    
    #normalize image
    S = cv.normalize(src.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    
    for cont in range(Num):
        #numpy.convolve(W, Y, mode='same')
        F = np.exp(-Alpha_F) * F + V_F * signal.convolve2d(Y, W, mode='same') + S
        L = np.exp(-Alpha_L) * L + V_L * signal.convolve2d(Y, M, mode='same')
        U = F * (1 + Beta * L)
        T = np.exp(-Alpha_T) * T + V_T * Y
        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    return Y_AC

def hough_transform(input_image, app):
    img = cv2.imdecode(numpy.fromstring(input_image, numpy.uint8), cv2.IMREAD_UNCHANGED)

    Y_AC = pcnn(img)
    edges = cv.Canny((Y_AC*255).astype(np.uint8),100,100,apertureSize = 3)
    lines_pcnn = cv.HoughLines(edges,1,np.pi/180,350)
    os.remove(file_path)

    if lines_pcnn is not None:
        return lines_pcnn
    else:
        return None