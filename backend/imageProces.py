import cv2
import numpy as np
#from matplotlib import pyplot as plt
import glob
import csv
import os

def subgraygray (gray1, gray2):
    #catatan ukuran gray 1 dan 2 harus sama dan inten sitas gray2 berupa 0 atau 255 (treshold)
    row, col = gray2.shape
    output = np.zeros((row,col,1), np.uint8)
    for i in range(0,row):
        for j in range(0,col):
            if int(gray1[i,j])-int(gray2[i,j]) < 0 :
                output.itemset((i,j,0),0)
            else:
                output.itemset((i,j,0),int(gray1[i,j])-int(gray2[i,j]))
    return output

def subrgbgray(rgb,treshold):
    row, col , raw = rgb.shape
    #print row*col
    output = np.zeros((row,col,3), np.uint8)
    for i in range(0,row):
        for j in range(0,col):
            if treshold[i,j] != 255:
                output.itemset((i,j,0),0)
                output.itemset((i,j,1),0)
                output.itemset((i,j,2),0)
            else:
                output[i,j]=rgb[i,j]
    return output

os.chdir('/home/adhan/Projek/sisdas-API/backend/temp')
tomat =  img = cv2.imread('img.JPG')
data=[]
kernel24 = np.ones((24, 24), np.uint8)
kernel3 = np.ones((3,3),np.uint8)

tomat = cv2.resize(tomat, (0,0), fx=0.1, fy=0.1)
    #cv2.imshow(imgname,tomat)
b,g,r = cv2.split( tomat )
tomat_segmented = cv2.subtract(g,b)
    # ret, tomat_segmented = cv2.threshold(tomat_segmented, 63,255,cv2.THRESH_BINARY)
ret, tomat_segmented = cv2.threshold(tomat_segmented, 11,255,cv2.THRESH_BINARY)
tomat_segmented = cv2.morphologyEx(tomat_segmented, cv2.MORPH_CLOSE, kernel24)
tomat_segmented = subrgbgray(tomat, tomat_segmented)
    #cv2.imshow(imgname,tomat_segmented)
    #HSV
rhsv = cv2.cvtColor(tomat_segmented, cv2.COLOR_BGR2HSV)

    # h,s,v = cv2.split(rhsv)
    # cv2.imshow(imgname+"HSV", s)
row, col, ch=tomat_segmented.shape

    #RGB &HSV tiap pixel
blue=0
red=0
green=0
hue=0
sat=0
val=0
for i in range(0, row):
    for j in range(0, col):
        b, g, r = tomat_segmented[i,j]
        h, s, v = rhsv[i,j]
        blue=blue+b
        green=green+g
        red=red+r
        hue=hue+h
        sat=sat+s
        val=val+v


b=blue/(row*col)
g=green/(row*col)
r=red/(row*col)
h=hue/(row*col)
s=sat/(row*col)
v=val/(row*col)
    # print(b)
    # print(g)
    # print(r)
    # x=[b,g,r]
x=[b,g,r,h,s,v]
data.append(x)



    #cv2.imwrite(imgname, tomat_segmented)
cv2.waitKey()

# a = np.asarray([[b,g,r]])
# a.tofile('datatomats.csv',sep=',',format='%10.5f')

myFile=open('predict.csv','wb')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(data)


cv2.destroyAllWindows()

