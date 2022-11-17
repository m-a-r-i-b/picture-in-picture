from concurrent.futures import ThreadPoolExecutor
from math import floor
import math
from multiprocessing import Pool
from functools import partial
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import time
from scipy import spatial



def getSamples(samplesFolder,subPicDimensions):

    h,w = subPicDimensions
    pointList,imgList = [], []

    for file in os.listdir(samplesFolder):
        currFilePath = samplesFolder+"/"+file
        img = cv2.imread(currFilePath)
        img =  cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
        avg = np.array(img).mean(axis=(0,1)).astype(int)
        pointList.append(avg)
        imgList.append(img)

    return pointList,imgList



def getClosestImg(croppedImg,kdTree,imgList):

    queryPoint = np.array(croppedImg).mean(axis=(0,1)).astype(int)
    index = kdTree.query(queryPoint)[1]
    return imgList[index]





img = cv2.imread("base3.jpg")
# img =  cv2.resize(img, (768,1152), interpolation = cv2.INTER_AREA)

imgCopy = cv2.imread("base3.jpg")
# imgCopy =  cv2.resize(imgCopy, (768,1152), interpolation = cv2.INTER_AREA)


h,w,c = img.shape

# TOTAL_SUB_PICS = 600

H_SUB_PICS = 200
W_SUB_PICS = 150

SUB_PIC_H = floor(h/H_SUB_PICS)
SUB_PIC_W = floor(w/W_SUB_PICS)

print("=============== ",w,h," ===============")
print("=============== ",SUB_PIC_W,SUB_PIC_H," ===============")


SAMPLE_FOLDER = "./samplesResized"
pointList,imgList = getSamples(SAMPLE_FOLDER,(SUB_PIC_H,SUB_PIC_W))
kdTree = spatial.KDTree(pointList)


currH, currW = 0,0
count = 0

t1 = time.time()
while currH+SUB_PIC_H < h:

    while currW+SUB_PIC_W < w:

        currStart = currW,currH
        currEnd = currW+SUB_PIC_W, currH+SUB_PIC_H

        image = cv2.rectangle(img, currStart, currEnd, (0,0,255), 1)
        # cv2.imshow("test",image)
        
        # cropped = [y:y+h, x:x+w]
        croppedImg = image[currStart[1]:currEnd[1], currStart[0]:currEnd[0]]

        newImg = getClosestImg(croppedImg,kdTree,imgList)
        imgCopy[currStart[1]:currEnd[1], currStart[0]:currEnd[0]] = newImg

        # cv2.imshow("currCrop",croppedImg)
        # cv2.imshow("newImg",imgCopy)
        # cv2.waitKey(0)
        
        currW += SUB_PIC_W
        count += 1
        print("Done with samples : ",count)
        # if(count == 20):
        #     t2 = time.time()
        #     cv2.imshow("test",imgCopy)
        #     cv2.waitKey(0)
        #     print(t2-t1)


    currW = 0
    currH += SUB_PIC_H


cv2.imwrite("output.jpg",imgCopy)
cv2.waitKey(0)