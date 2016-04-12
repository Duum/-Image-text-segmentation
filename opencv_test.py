# -*- coding: utf-8 -*-
import cv2
import numpy as np
from scipy.cluster.vq import kmeans


def image_word_matrix(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray",gray)
    gray = np.float32(gray)
    # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    dst = cv2.cornerHarris(gray,2,3,0.04)
    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)
    # Threshold for an optimal value, it may vary depending on the image.
    img[dst>0.01*dst.max()]=[255,255,255]
    gray1 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow('dst',gray1)
    gray2 = (gray1-gray)
    #cv2.imshow("san",gray2)

    gray2 = np.int32(gray2)


    image_word_matrix = []
    for i in range(40):
        for j in range(150):
            if gray2[i,j]>10:
                image_word_matrix.append([i,j])
    return image_word_matrix


def image_cut_kmeans(word_matrix,n):
    data = np.vstack(word_matrix)
    centroids,_ = kmeans(data,4)
    return centroids

if __name__ == "__main__":
    file_path = "hubei_image/2.jpg"
    word_matrix = image_word_matrix(file_path)
    #for i in range(len(word_matrix)):
        #print word_matrix[i]
    a = image_cut_kmeans(np.asmatrix(word_matrix),6)

    b = []
    for i in range(len(a)):
        b.append(a[i][1])
    b.sort()
    img = cv2.imread(file_path)
    print img.shape[:2][1]
    for i in range(3):
        cut1 = (b[i] + b[i+1]) / 2
        if b[i]-6>0:
            bound_l = b[i]-14
        else:
            bound_l = b[i]
        img1 = img[:,bound_l:cut1]
        cv2.imshow(str(i),img1)

    bound_r = []
    for j in range(len(word_matrix)):
        bound_r.append(word_matrix[j][1])
    bound_r.sort()
    n = len(bound_r) - 1
    if bound_r[n] + 4 <= img.shape[:2][1]:
        r = bound_r[n] + 4
    else:
        r = bound_r[n]
    img1 = img[:,cut1:r]
    cv2.imshow(str(4),img1)



    cv2.waitKey()