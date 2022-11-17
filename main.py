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



def getSamples(samplesFolder,subPicDimensions,outputPicDimensions):

    h,w = subPicDimensions
    outputH, outputW = outputPicDimensions
    pointList,imgList,outputImgList = [], [], []

    for file in os.listdir(samplesFolder):
        currFilePath = samplesFolder+"/"+file
        img = cv2.imread(currFilePath)
        outputImg = cv2.resize(img, (outputW,outputH), interpolation = cv2.INTER_AREA) 
        img =  cv2.resize(img, (w,h), interpolation = cv2.INTER_AREA)
        avg = np.array(img).mean(axis=(0,1)).astype(int)
        pointList.append(avg)
        imgList.append(img)
        outputImgList.append(outputImg)

    return pointList,imgList,outputImgList



def getClosestImg(croppedImg,kdTree,outputImgList):

    queryPoint = np.array(croppedImg).mean(axis=(0,1)).astype(int)
    index = kdTree.query(queryPoint)[1]
    return outputImgList[index]





img = cv2.imread("base3.jpg")
# img =  cv2.resize(img, (768,1152), interpolation = cv2.INTER_AREA)

# imgCopy = cv2.imread("base3.jpg")
# imgCopy = np.zeros([100,100,3],dtype=np.uint8)

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
OUTPUT_SUB_PIC_DIMENSIONS = 192,128
pointList,imgList,outputImgList = getSamples(SAMPLE_FOLDER,(SUB_PIC_H,SUB_PIC_W),OUTPUT_SUB_PIC_DIMENSIONS)
kdTree = spatial.KDTree(pointList)


OUTPUT_PIC_DIMENSIONS = OUTPUT_SUB_PIC_DIMENSIONS[0]*H_SUB_PICS, OUTPUT_SUB_PIC_DIMENSIONS[1]*W_SUB_PICS

imgCopy = np.zeros([OUTPUT_PIC_DIMENSIONS[0],OUTPUT_PIC_DIMENSIONS[1],3],dtype=np.uint8)

print(imgCopy.shape)

currH, currW = 0,0
count = 0

t1 = time.time()
while (currH+1)*OUTPUT_SUB_PIC_DIMENSIONS[0] < OUTPUT_PIC_DIMENSIONS[0]:

    while (currW+1)*OUTPUT_SUB_PIC_DIMENSIONS[1] < OUTPUT_PIC_DIMENSIONS[1]:

        currStart = currW*SUB_PIC_W,currH*SUB_PIC_H
        currEnd = (currW+1)*SUB_PIC_W, (currH+1)*SUB_PIC_H

        image = cv2.rectangle(img, currStart, currEnd, (0,0,255), 1)
        # cv2.imshow("test",image)
        
        # cropped = [y:y+h, x:x+w]
        croppedImg = image[currStart[1]:currEnd[1], currStart[0]:currEnd[0]]

        newImg = getClosestImg(croppedImg,kdTree,outputImgList)
        

        currOutStart = currW*OUTPUT_SUB_PIC_DIMENSIONS[1],currH*OUTPUT_SUB_PIC_DIMENSIONS[0]
        currOutEnd = (currW+1)*OUTPUT_SUB_PIC_DIMENSIONS[1], (currH+1)*OUTPUT_SUB_PIC_DIMENSIONS[0]

        imgCopy[currOutStart[1]:currOutEnd[1], currOutStart[0]:currOutEnd[0]] = newImg

        # cv2.imshow("currCrop",croppedImg)
        # test = cv2.resize(imgCopy, (500,800), interpolation = cv2.INTER_AREA)
        # cv2.imshow("newImg",test)
        # cv2.waitKey(0)
        
        currW += 1
        count += 1
        print("Done with samples : ",count)
        # if(count == 20):
        #     t2 = time.time()
        #     cv2.imshow("test",imgCopy)
        #     cv2.waitKey(0)
        #     print(t2-t1)


    currW = 0
    currH += 1


cv2.imwrite("output.jpg",imgCopy)
cv2.waitKey(0)