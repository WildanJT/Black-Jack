import os
import cv2
import numpy as np


def CreateDataset(dset = 'Suit', BKG_THRESH= 100, dim= 100):
    suit_dir = 'Cards/' + dset + '/'
    suitList = os.listdir(suit_dir)

    i = 0 

    for img_dir in suitList:
        img = cv2.imread(suit_dir + img_dir)
        img = cv2.resize(img, (70, dim))
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        img_w, img_h = np.shape(img)[:2]
        thresh_level = 0 + BKG_THRESH
        
        retval, thresh = cv2.threshold(blur, thresh_level, 255, cv2.THRESH_BINARY)
        reverse_img = cv2.bitwise_not(thresh)

        # Writting new image at new directory
        temp = os.getcwd()
        
        os.chdir(os.getcwd() + '/Dataset/New')

        cv2.imwrite(f'{os.path.splitext(suitList[i])[0]}.jpg', reverse_img)
        print("Image", os.path.splitext(img_dir)[0], "have been processed to new directory....")
        
        os.chdir(temp)
        
        i = i + 1


""" Function Call """
# 'Rank', 70, 125
CreateDataset()

