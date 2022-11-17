from math import floor
import cv2
import numpy as np


img = cv2.imread("tennis.jpg")
# img = np.zeros([500,500,3])

h,w,c = img.shape


TOTAL_SUB_PICS = 100

H_SUB_PICS = TOTAL_SUB_PICS/2
W_SUB_PICS = TOTAL_SUB_PICS/2


SUB_PIC_H = floor(h/H_SUB_PICS)
SUB_PIC_W = floor(w/W_SUB_PICS)


currH, currW = 0,0

while currH+SUB_PIC_H < h:

    print("-"*50)

    while currW+SUB_PIC_W < w:

        currStart = currW,currH
        currEnd = currW+SUB_PIC_W, currH+SUB_PIC_H

        print("Start = ",currStart)
        print("END   = ",currEnd)
        
        image = cv2.rectangle(img, currStart, currEnd, (0,0,255), 1)
        cv2.imshow("test",image)
        
        # cropped = [y:y+h, x:x+w]
        cropped = image[currStart[1]:currEnd[1], currStart[0]:currEnd[0]]

        cv2.imshow("currCrop",cropped)
        cv2.waitKey(0)
        
        currW += SUB_PIC_W

    currW = 0
    currH += SUB_PIC_H


# print("h,w = ",h,w)
# print("sh,sw = ",SUB_PIC_H,SUB_PIC_W)
# cv2.imshow("test",img)
# cv2.waitKey(0)