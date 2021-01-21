'''
Credits: https://github.com/arnoldcvl/pcnn_functions
'''
import sys
import numpy as np
import cv2 as cv
from scipy import signal
import math
import os

from kht import kht
from werkzeug.utils import secure_filename
from app import app
from sklearn.cluster import KMeans

def dist_gausian_kernel(link_arrange):
    kernel = np.zeros((link_arrange, link_arrange), np.float)

    center_x = round(link_arrange/2)
    center_y = round(link_arrange/2)
    for i in range(link_arrange):
        for j in range(link_arrange):
            if (i == center_x) and (j == center_y):
                kernel[i,j] = 1
            else:
                kernel[i,j] = 1/(math.sqrt(((i)-center_x)**2+((j)-center_y)**2))
    return kernel


def get_intensity(source):
    source = cv.normalize(source.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    R = source[:,:,0]
    G = source[:,:,1]
    B = source[:,:,2]
    intensity = np.divide(R+G+B, 3)
    
    s = cv.normalize(intensity.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)
    return s


def pcnn(source, Alpha_L=0.1, Alpha_T=0.5, V_T=1.0, W=dist_gausian_kernel(9), Beta=0.1, T_extra=63, Num=10):
    dim = source.shape

    F = np.zeros( dim, np.float)
    L = np.zeros( dim, np.float)
    Y = np.zeros( dim, np.float)
    T = np.ones( dim, np.float) + T_extra
    Y_AC = np.zeros( dim, np.float)

    for cont in range(Num):
        L = Alpha_L * signal.convolve2d(Y, W, mode='same')
        U = source * (1 + Beta * L)
        YC = 1 - Y      
        T = T - Alpha_T
        T = ((Y*T)*V_T) + (YC*T)
        Y = (U>T).astype(np.float)
        Y_AC = Y_AC + Y
    
    Y_AC = cv.normalize(Y_AC.astype('float'), None, 0.0, 1.0, cv.NORM_MINMAX)

    return Y, Y_AC


def init_kht(source, K_cluster_min_size=10, K_kernel_min_height=0.002, K_cluster_min_deviation=2.0, K_delta=0.5, K_n_sigmas=2):
    source_copy = source.copy()
    source_copy = (source_copy*255).astype(np.uint8)

    kernel = np.ones((5,5),np.float32)
    filter_result = cv.filter2D(source_copy, cv.CV_8U, kernel)
    edges_result = cv.Canny(filter_result, 80, 200)
    edges_copy = edges_result.copy()

    k = kht(edges_copy, 
            cluster_min_size=K_cluster_min_size, 
            cluster_min_deviation=K_cluster_min_deviation, 
            kernel_min_height=K_kernel_min_height, 
            delta=K_delta, 
            n_sigmas=K_n_sigmas)
    
    return k, filter_result


def showLines_kht(lines, source, lines_count=0):
    if lines_count==0: lines_count=len(lines)

    if(len(source.shape)>2): height, width, _ = source.shape
    else: height, width = source.shape

    source_copy = source.copy()

    for (rho, theta) in lines[:lines_count]:
        theta = math.radians(theta)
        cos_theta, sin_theta = math.cos(theta), math.sin(theta)

        h2 = height/2
        w2 = width/2
        if sin_theta != 0:
            one_div_sin_theta = 1 / sin_theta
            x = (int(round(0)) , int(round(h2 + (rho + w2 * cos_theta) * one_div_sin_theta)))
            y = (int(round(w2+w2)) , int(round(h2 + (rho - w2 * cos_theta) * one_div_sin_theta)))
        else:
            x = (int(round(w2 + rho)), int(round(0)))
            y = (int(round(w2 + rho)), int(round(h2+h2)))
        cv.line(source_copy, x, y, (120,0,255), 2, cv.LINE_AA)

    return source_copy


def init_kmeans(lines, interations=10, qtd_clusters=10):
    X = np.asarray(lines, dtype=np.float32)
    qtd_180 = X[:,1]>178
    qtd_0 = X[:,1]<2

    if np.count_nonzero(qtd_180) > np.count_nonzero(qtd_0):
        X[qtd_0, 1]=179.5
    else:
        X[qtd_180, 1]=0
    
    k_means = KMeans(init='k-means++', n_clusters=qtd_clusters, n_init=interations)
    k_means.fit(X)
    k_means_labels = k_means.labels_
    k_means_cluster_centers = k_means.cluster_centers_
    k_means_labels_unique = np.unique(k_means_labels)

    return k_means.cluster_centers_

def get_lines(input_image, app):
    img = cv.imread(save_image(input_image))

    i_img = get_intensity(img)
    result, result2 = pcnn(i_image,T_extra=43, Num=10)
    lines,filt = init_kht(result)
    k_lines = init_kmeans(lines)
    
    if k_lines is not None:
        print(type(k_lines))
        print(k_lines)
        return k_lines
    else:
        return "No Line Detected"


def save_image(img_file):
    filename = secure_filename(img_file.filename)
    img_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

