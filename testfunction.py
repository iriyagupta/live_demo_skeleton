import cv2
import os
import re
import numpy as np
import glob
import random


def inpaint(image):
    print(image)
    portion_size=(64,64,3)   #size of the patch taken
    img = cv2.imread(image,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    rgb_image = img.copy()
    image_size=img.shape
    #creating rectangular masks out of the sample images
    x1 = random.randint(0,image_size[0]-portion_size[0]-1)
    y1 = random.randint(0,image_size[1]-portion_size[1]-1)
    x2, y2 = x1+portion_size[0]-1, y1+portion_size[1]-1
    x2, y2 = x1+portion_size[0]-1, y1+portion_size[1]-1
    img[x1:x2,y1:y2]=[255,255,255]  #assigning the values to black
    difference = img - rgb_image
    difference[x1:x2,y1:y2] = [255,255,255]
    # cv2.imread("difference")
    difference = cv2.cvtColor(difference,cv2.COLOR_RGB2GRAY)
    dst = cv2.inpaint(rgb_image,difference,3,cv2.INPAINT_TELEA)
    new = cv2.imwrite(os.path.join( "static/tampered", "output.jpg" ), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))
    #scv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dst
