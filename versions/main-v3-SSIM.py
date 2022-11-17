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


def getSamples(samplesFolder):

    samples = []
    for file in os.listdir(samplesFolder):
        currFilePath = samplesFolder+"/"+file
        img = cv2.imread(currFilePath)
        samples.append(img)

    return samples



def getClosestImgSeq(croppedImg,samples):

    h,w,c = croppedImg.shape
    
    bestMatchVal = -math.inf
    bestMatchImg = None

    for sample in samples:
        resizedSample =  cv2.resize(sample, (w,h), interpolation = cv2.INTER_AREA)
        currSimilarity = ssim(croppedImg, resizedSample, multichannel=True)

        if(currSimilarity > bestMatchVal):
            bestMatchVal = currSimilarity
            bestMatchImg = resizedSample

    return bestMatchVal,bestMatchImg



def getImgToSampleSimilarity(croppedImg,sample):
    h,w,c = croppedImg.shape
    resizedSample =  cv2.resize(sample, (w,h), interpolation = cv2.INTER_AREA)
    currSimilarity = ssim(croppedImg, resizedSample, multichannel=True)
    return [currSimilarity,resizedSample]



def getClosestImgMultiProcess(croppedImg,samples):

    with Pool(5) as p:
        similaritiesAndImages = p.map(partial(getImgToSampleSimilarity, croppedImg), samples)

    similaritiesAndImages = sorted(similaritiesAndImages)
    bestMatchVal,bestMatchImg = similaritiesAndImages[-1]

    return bestMatchVal,bestMatchImg


def getClosestImgMultiThread(croppedImg,samples):

    with ThreadPoolExecutor(5) as executor:
        similaritiesAndImages = executor.map(partial(getImgToSampleSimilarity, croppedImg), samples)

    similaritiesAndImages = sorted(similaritiesAndImages)
    bestMatchVal,bestMatchImg = similaritiesAndImages[-1]

    return bestMatchVal,bestMatchImg





img = cv2.imread("base2.jpg")
# img =  cv2.resize(img, (768,1152), interpolation = cv2.INTER_AREA)

imgCopy = cv2.imread("base2.jpg")
# imgCopy =  cv2.resize(imgCopy, (768,1152), interpolation = cv2.INTER_AREA)


h,w,c = img.shape

# TOTAL_SUB_PICS = 600

H_SUB_PICS = 120
W_SUB_PICS = 90

SUB_PIC_H = floor(h/H_SUB_PICS)
SUB_PIC_W = floor(w/W_SUB_PICS)

print("=============== ",w,h," ===============")
print("=============== ",SUB_PIC_W,SUB_PIC_H," ===============")

SAMPLE_FOLDER = "./samplesResized"


samples = getSamples(SAMPLE_FOLDER)

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

        val, newImg = getClosestImgSeq(croppedImg,samples)
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