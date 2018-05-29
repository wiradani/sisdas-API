from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import glob
import csv
import os
from app import *

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


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


kernel24 = np.ones((24, 24), np.uint8)
kernel3 = np.ones((3,3),np.uint8)

#imgnames=sorted(glob.glob('Kelas1/*.jpg'))
#data=[]
#for imgname in imgnames:
def sizeObj(filename):
	os.chdir('/home/adhan/Projek/sisdas-API/backend/temp')
	tomat = cv2.imread(filename)
	tomat = cv2.resize(tomat, (0,0), fx=0.1, fy=0.1)
			#cv2.imshow(imgname,tomat)
	b,g,r = cv2.split( tomat )
	tomat_segmented = cv2.subtract(r,g)
			# ret, tomat_segmented = cv2.threshold(tomat_segmented, 63,255,cv2.THRESH_BINARY)
	ret, tomat_segmented = cv2.threshold(tomat_segmented, 9,255,cv2.THRESH_BINARY)
	tomat_segmented = cv2.morphologyEx(tomat_segmented, cv2.MORPH_CLOSE, kernel3)
	tomat_segmented = subrgbgray(tomat, tomat_segmented)


	gray = cv2.cvtColor(tomat_segmented, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (7, 7), 0)

	edged = cv2.Canny(gray, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = None



	for c in cnts:

		if cv2.contourArea(c) < 100:
			continue

		orig = tomat_segmented.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

		(tl, tr, br, bl) = box
		(tltrX, tltrY) = midpoint(tl, tr)
		(blbrX, blbrY) = midpoint(bl, br)

		(tlblX, tlblY) = midpoint(tl, bl)
		(trbrX, trbrY) = midpoint(tr, br)

		cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
		cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

		cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
			(255, 0, 255), 2)
		cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
			(255, 0, 255), 2)

		dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
		dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

		if pixelsPerMetric is None:
			pixelsPerMetric = dB / 3.779528

		dimA = dA / pixelsPerMetric
		dimB = dB / pixelsPerMetric

		cv2.putText(orig, "{:f}".format(dimA),
			(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)
		cv2.putText(orig, "{:f}".format(dimB),
			(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
			0.65, (255, 255, 255), 2)

		
		dimA=dimA*2.54
		dimB=dimB*2.54
		print(dimA, dimB)
		cv2.waitKey(0)

		#hitung berat objek
		diameter=dimA
		r=diameter/2
		vol=(4/3)*3.14*r*r*r
		weight=vol*1.01
		print(r)
		print(diameter)

	
	cv2.destroyAllWindows()
	return weight



