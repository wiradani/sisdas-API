import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
import csv
import random

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

imgnames=sorted(glob.glob('Kelas 1/*.JPG'))
data=[]
colb=[]
for imgname in imgnames:
	## penjelasan -> membuat kernel dengan size 24x24
    kernel24 = np.ones((24, 24), np.uint8)

	## penjelasan -> membaca image dengan fungsi imread
    tomat = cv2.imread(imgname)

	## resize ukuran citra tomato dengan pengskalaan menjadi 0.1x0.1
    tomat = cv2.resize(tomat, (0,0), fx=0.1, fy=0.1)

	## menampilkan citra yang telah di resize
    cv2.imshow(1_img_resize,tomat)

	## cv2.split membagi image menjadi red greed blue channel yang terpisah
    b,g,r = cv2.split( tomat )

	## melakukan operasi pengurangan channel red dengan blue
    tomat_segmented = cv2.subtract(r,b)
   cv2.imshow(2_img_segmented, tomat_segmented)

	## image di threshold dengan binnary -> image menjadi hitam putih
    ret, tomat_segmented = cv2.threshold(tomat_segmented, 15,255,cv2.THRESH_BINARY)
cv2.imshow(3_img_thresh_binary,tomat_segmented)

	## dengan morphology melakukan fungsi close (proses dilasi diiukuti erosi)
## closing berfungsi untuk mengisi lubang kecil (noise) pada objek / menghaluskan objek
    tomat_segmented = cv2.morphologyEx(tomat_segmented, cv2.MORPH_CLOSE, kernel24)
cv2.imshow(4_img_closing,tomat_segmented)

## masuk fungsi untuk memasukan img tomat pada bagian putih di img tomat_segmented
    tomat_segmented = subrgbgray(tomat, tomat_segmented)
cv2.imshow(5_img_merge,tomat_segmented)

    #HSV -> convert bgr to hsv
    rhsv = cv2.cvtColor(tomat_segmented, cv2.COLOR_BGR2HSV)
cv2.imshow(6_img_hsv,rhsv)

# split img jadi channel h,s,v
    h,s,v = cv2.split(rhsv)

    #cv2.imshow(imgname+"HSV", s)
    row, col, ch=tomat_segmented.shape

    #RGB &HSV tiap pixel
    blue=0
    red=0
    green=0
    hue=0
    sat=0
    val=0
    hit=0 #hitung hitam
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
            if (g==0 & r==0 & b==0 ):
                hit=hit+1


    # b=blue/(row*col)
    # g=green/(row*col)
    # r=red/(row*col)
    # h=hue/(row*col)
    # s=sat/(row*col)
    # v=val/(row*col)
    hi=hit/(row*col)
    to=(1-hi)*100
    # print(b)
    # print(g)
    # print(r)
    #berat
    berat=random.randrange(27,31,1)
    #csv warna
    #x=[b,g,r,h,s,v]
    #data.append(x)
    #csv berat
    y=[to,berat]
    data.append(y)
    # print(hit)
    # print(hi)




    cv2.waitKey()

# a = np.asarray([[b,g,r]])
# a.tofile('kelas1.csv',sep=',',format='%10.5f')

# myFile=open('kelas1.csv','w', newline='')
# with myFile:
#     writer = csv.writer(myFile)
#     writer.writerows(data)
name=["da_tomat", "berat"]
colb.append(name)
myFile=open('berat1.csv','w', newline='')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(colb)
    writer.writerows(data)

#histogram dengan RGB
# color = ('b','g','r')
# for i,cols in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     print(cols)
#     for j, his in enumerate(histr):
#         print(j)
#         print(histr[j])
#     plt.plot(histr,color = cols)
#     plt.xlim([0,256])
# plt.show()
# r,g,b=cv2.split(img)
# print(r)
# print(g)
# print(b)

cv2.destroyAllWindows()


