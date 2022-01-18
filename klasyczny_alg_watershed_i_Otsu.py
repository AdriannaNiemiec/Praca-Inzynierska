from posixpath import split
import cv2  #cv2 will read images
import os   #os will read files
import numpy as np  #this is used to store images as array
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from skimage import measure, color, io
from scipy import ndimage
import argparse
import imutils


def Hist(image):
    H = np.zeros(shape=(256,1))
    shape = image.shape
    for i in range(shape[0]):  
        for j in range(shape[1]):   
            k=image[i,j]   
            H[k,0]=H[k,0]+1
    return H

def inverse(image):
    shape = image.shape
    for i in range(shape[0]):  
        for j in range(shape[1]):  
            if image[i,j]==0:
                image[i,j]=255
            elif image[i,j]==255:
                image[i,j]=0
    return image

def threshold(img):
    image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(image,(5,5),0)
    otsu_threshold, image_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    histx = Hist(image_thresh)
    if histx[0]<histx[255]:
        image_thresh = inverse(image_thresh)
    return image_thresh

def separation(img):
    image_thresh = threshold(img)
    #remove noise - opening
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(image_thresh, cv2.MORPH_OPEN,kernel, iterations = 2)
    ### Watershed Segmentation
    # sure background
    sure_bg = cv2.dilate(opening, kernel,iterations=3)
    # sure foreground
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret2, sure_fg = cv2.threshold(dist_transform,0.4*dist_transform.max(),255,0)
    # Unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    # markers
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers+10
    markers[unknown==255] = 0
    # watershed 
    markers = cv2.watershed(img,markers)
    # segment nuclei
    img[markers == -1] = [0,255,255]
    image_result = [image_thresh]
    counter=0
    for label in np.unique(markers):
        if counter == 0 or counter ==1:
            counter+=1
            continue
        if label == 0:
            continue
        mask = np.zeros(img.shape, dtype="uint8")
        mask[markers == label] = 255
        image_result.append(mask) 
    return image_result
   

def display(img, name):
    image_result = separation(img)
    #saveMasks(image_result, name)
    for i in image_result:
        cv2.imshow("mask",i)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return image_result

def saveMasks(img_result, name):
    path1 = "Y_train/" + name
    os.mkdir(path1) #create path
    for index in range(1, len(img_result)):
        path2 = r'Y_train/' + name + '/mask' + str(index) + '.png'
        io.imsave(path2, img_result[index])
    return img_result


dataset = os.popen('dir .\dataset').read()
splitted = dataset.splitlines()
splitted = splitted[7:len(splitted)-2]
x=0
for i in splitted:
    splitted[x] = i[36:len(i)]
    x+=1

for i in splitted:
    path = r'dataset/' + i + '/images/' + i + '.png'
    img = cv2.imread(path)
    img = display(img, i) 
