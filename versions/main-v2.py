from math import floor
import math
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim


def getSamples(samplesFolder):

    samples = []
    for file in os.listdir(samplesFolder):
        currFilePath = samplesFolder+"/"+file
        img = cv2.imread(currFilePath)
        samples.append(img)

    return samples

def getClosestImg(croppedImg,samples):

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





img = cv2.imread("tennis.jpg")
imgCopy = cv2.imread("tennis.jpg")

h,w,c = img.shape

TOTAL_SUB_PICS = 100

H_SUB_PICS = TOTAL_SUB_PICS/2
W_SUB_PICS = TOTAL_SUB_PICS/2

SUB_PIC_H = floor(h/H_SUB_PICS)
SUB_PIC_W = floor(w/W_SUB_PICS)

SAMPLE_FOLDER = "./samples"


samples = getSamples(SAMPLE_FOLDER)

currH, currW = 0,0

while currH+SUB_PIC_H < h:

    while currW+SUB_PIC_W < w:

        currStart = currW,currH
        currEnd = currW+SUB_PIC_W, currH+SUB_PIC_H

        image = cv2.rectangle(img, currStart, currEnd, (0,0,255), 1)
        cv2.imshow("test",image)
        
        # cropped = [y:y+h, x:x+w]
        croppedImg = image[currStart[1]:currEnd[1], currStart[0]:currEnd[0]]

        val, newImg = getClosestImg(croppedImg,samples)
        imgCopy[currStart[1]:currEnd[1], currStart[0]:currEnd[0]] = newImg

        cv2.imshow("currCrop",croppedImg)
        cv2.imshow("newImg",imgCopy)
        cv2.waitKey(0)
        
        currW += SUB_PIC_W

    currW = 0
    currH += SUB_PIC_H

