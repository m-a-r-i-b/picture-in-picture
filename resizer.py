import cv2
import os



samplesFolder = "./samples"
resizedSamplesFolder = "./samplesResized"

for file in os.listdir(samplesFolder):
    currFilePath = samplesFolder+"/"+file
    img = cv2.imread(currFilePath)

    h,w,c = img.shape
    newH,newW = round(h/10),round(w/10)
    resizedSample =  cv2.resize(img, (newW,newH), interpolation = cv2.INTER_AREA)
    print("new path = ",resizedSamplesFolder+"/"+file)
    cv2.imwrite(resizedSamplesFolder+"/"+file,resizedSample)


