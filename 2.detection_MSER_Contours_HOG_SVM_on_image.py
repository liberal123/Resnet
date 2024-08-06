import glob
import pandas as pd  # as means that we use pandas library short form  as pd
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt # matplotlib is big library, we are just calling pyplot function 
from skimage.feature import hog #We are calling only hog  
from sklearn.externals import joblib # Calling the joblib function from sklearn, use for model saving 
                                     # and loading.
import matplotlib.pyplot as pyplot
import sys
np.warnings.filterwarnings('ignore')

# Loading the mode into same name
pca = torch.load('F:\python project\交通标识识别项目1\model\pca.pkl')
classifier = torch.load('F:\python project\交通标识识别项目1\model\svm.pkl')


def imadjust(x,a,b,c,d,gamma=1):
    # Similar to imadjust in MATLAB.
    # Converts an image range from [a,b] to [c,d].
    # The Equation of a line can be used for this transformation:
    #   y=((d-c)/(b-a))*(x-a)+c
    # However, it is better to use a more generalized equation:
    #   y=((x-a)/(b-a))^gamma*(d-c)+c
    # If gamma is equal to 1, then the line equation is used.
    # When gamma is not equal to 1, then the transformation is not linear.

    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y
def image_fill(Binary_image):
    # Mask used to flood filling.
    im_th=Binary_image.astype('uint8').copy()#RGB图像的值是处于0-255之间的，为了更好的处理图像，通常会将图像值转变到0-1之间
    #这个处理的过程就是图像的uint8类型转变为float类型过程  uint8是专门用于存储各种图像的（包括RGB，灰度图像等），范围是从0–255
    h, w = im_th.shape[:2]
    im_floodfill = im_th.copy()
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 1); # 泛洪算法——floodFill函数原型

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv
    im_out[im_out==254]=0
    #测试1
    # cv2.imshow('test', im_out)
    # cv2.waitKey(0)
    # cv2.destroyWindow('test')

    # print(type(im_out))#<class 'numpy.ndarray'>
    return im_out

def cnts_find(binary_image_blue,binary_image_red):
    cont_Saver=[]
    
    cnts, hierarchy  = cv2.findContours(binary_image_blue.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
    # cnts_ny=np.array(cnts) 错了
    #测试 cnts 轮廓  M*N  M是轮廓个数  N是每个轮廓的点
    # -----------------画图测试-----------------------
    # pyplot.imshow(cnts_ny)
    # pyplot.show() #要加这一行才能画出来
    print(cnts[0]) #[[[ 54 147]]]
    # print(len(cnts)) 83
    # print(type(cnts)) <class 'tuple'>
    # cv2.imshow('test', cnts)
    # cv2.waitKey(0)
    # cv2.destroyWindow('test')
    for d in cnts:
         if cv2.contourArea(d)>700:
                (x, y, w, h) = cv2.boundingRect(d)
                if ((w/h)<1.21 and (w/h)>0.59 and w>20):
                    cont_Saver.append([cv2.contourArea(d),x, y, w, h])
    
    cnts, hierarchy = cv2.findContours(binary_image_red.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#finding contours of conected component
    for d in cnts: #d是取出来所有轮廓中的一个
         if cv2.contourArea(d)>700:
                (x, y, w, h) = cv2.boundingRect(d)
                if ((w/h)<1.21 and (w/h)>0.59 and w>20):
                    cont_Saver.append([cv2.contourArea(d),x, y, w, h])

    return cont_Saver
#-------------------------测试---------------------------------
image_path=r'F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Testing\00038\00344_00000.ppm'
# image_path=r'F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Testing\00021\00047_00001.ppm'
# image_path=r'F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Testing\00035\00151_00002.ppm' # Tested image path
# image_path=sys.argv[1]
print ('Reading Image from ',image_path)

img = cv2.imread(image_path)
img_rgb=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_rgb[:,:,0] = cv2.medianBlur(img_rgb[:,:,0],3) #applying median filter to remove noice 3是核的大小
img_rgb[:,:,1] = cv2.medianBlur(img_rgb[:,:,1],3) #applying median filter to remove noice
img_rgb[:,:,2] = cv2.medianBlur(img_rgb[:,:,2],3) #applying median filter to remove noice

arr2=img_rgb.copy()
arr2 = cv2.normalize(arr2.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

imgr=arr2[:,:,0]
imgg=arr2[:,:,1]
imgb=arr2[:,:,2]
#-----------------画图测试-----------------------
# pyplot.imshow(imgr)
# pyplot.imshow(imgg)
# pyplot.imshow(imgb)
# pyplot.show() #要加这一行才能画出来
# Calling the imgadujust function
imgr=imadjust(imgr,imgr.min(),imgr.max(),0,1)  #把值限定在0-1之间
imgg=imadjust(imgg,imgg.min(),imgg.max(),0,1)
imgb=imadjust(imgb,imgb.min(),imgb.max(),0,1)
#4Normalize intensity of the red channel
Cr = np.maximum(0,np.divide(np.minimum((imgr-imgb),(imgr-imgg)),(imgr+imgg+imgb)))
Cr[np.isnan(Cr)]=0 #np.isnan(x)函数可以判断x是否为空值，然后输出布尔类型的变量
#Normalize intensity of the blue channel
Cb = np.maximum(0,np.divide((imgb-imgr),(imgr+imgg+imgb)))
Cb[np.isnan(Cb)]=0

[rows,cols]=img[:,:,1].shape
#Red color, normalization then thresholding it as 1
sc=(cv2.normalize(Cr.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int') #指定将图片的值放缩到 0-255 之间 范围缩放函数
# 注意 这里是对比度归一化之后的图像拿去进行MSER 就会把那两张红蓝对比度中的白色提取出来
mser = cv2.MSER_create(min_area=100,max_area=10000)

regions, _ = mser.detectRegions(sc.astype('uint8')) #用MSER把像素相同的区域提取出来  这个是红色的
#-----------------画图测试-----------------------
# pyplot.imshow(regions[0])
# pyplot.show() #要加这一行才能画出来
# cv2.imshow('test', regions[0]) #这里是二维的 CV2.imshow只能展示3维？
# cv2.waitKey(0)
# cv2.destroyWindow('test')
# print(regions[0])
# print(regions)
# print(regions[0][0]) [ 53 153]这是像素点 一个通道所以是二维的
# print(len(regions[0])) 731 这是第一个区域的像素点个数
# print(len(regions[1])) 5447 这是第二个区域的像素点个数
# print(type(regions[0]))<class 'numpy.ndarray'>
# print(type(regions)) #<class 'tuple'>
# print(len(regions)) 109
BMred=np.zeros((rows,cols))
if len(regions)>0:
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            BMred[regions[i][j][1],regions[i][j][0]]=1 #在有像素的地方把像素值替换为1 有些地方没有提取出来 就是0
#-----------------画图测试-----------------------
# pyplot.imshow(BMred)
# pyplot.show() #要加这一行才能画出来
# print(BMred)
# print(len(BMred))
# print(BMred.shape)
#Blue color, normalization then thresholding it as 1
sb=(cv2.normalize(Cb.astype('float'), None, 0, 255, cv2.NORM_MINMAX)).astype('int')
mser = cv2.MSER_create(min_area=100,max_area=10000)
regions, _ = mser.detectRegions(sb.astype('uint8'))
BMblue=np.zeros((rows,cols))
if len(regions)>0:
    for i in range(len(regions)):
        for j in range(len(regions[i])):
            BMblue[regions[i][j][1],regions[i][j][0]]=1
#-----------------画图测试-----------------------
# pyplot.imshow(BMblue)
# pyplot.show() #要加这一行才能画出来
        
img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# hsv color 

## Hsv range for red
s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
#-----------------画图测试-----------------------
# pyplot.imshow(s)
# pyplot.show() #要加这一行才能画出来
# # V通道
# pyplot.imshow(v)
# pyplot.show() #要加这一行才能画出来
# print(s)
# print(s[s<0.5]) #[0.11764706 0.14117647 0.17254902 ... 0.12156863 0.16078431 0.14509804]
# print(s<0.5)
# [[ True  True  True ...  True  True  True]
#  [ True  True  True ...  True  True  True]
#  [ True  True  True ...  True  True  True]
#  ...
#  [ True  True  True ...  True  True  True]
#  [ True  True  True ...  True  True  True]
#  [ True  True  True ...  True  True  True]]
s[s<0.5]=0

s[s>0.65]=0 #这个范围的都变成0
s[s>0]=1 #只有这个范围是1
# print(s)
v[v<0.2]=0
v[v>0.75]=0
v[v>0]=1
redmask=np.multiply(s,v) #掩码

## Hsv range for blue
s=cv2.normalize(img_hsv[:,:,1].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
v=cv2.normalize(img_hsv[:,:,2].astype('float'), None, 0, 1, cv2.NORM_MINMAX)
s[s<0.45]=0
s[s>0.80]=0
s[s>0]=1

v[v<0.35]=0
v[v>1]=0
v[v>0]=1
# # #--------------------------------------------------------------------
# pyplot.imshow(s)
# pyplot.show() #要加这一行才能画出来
# # V通道
# pyplot.imshow(v)
# pyplot.show() #要加这一行才能画出来
# #--------------------------------------------------------------------
# pyplot.imshow(redmask)
# pyplot.show() #0
bluemask=np.multiply(s,v) #1
# pyplot.imshow(bluemask)
# pyplot.show() #要加这一行才能画出来 1
#
# # Taking the common part that is in both.
BMred_mask=np.multiply(BMred,redmask)
BMblue_mask=np.multiply(BMblue,bluemask)
# pyplot.imshow(BMred_mask)
# pyplot.show() #要加这一行才能画出来 2 这个没有完整地轮廓
# pyplot.imshow(BMblue_mask)
# pyplot.show() #要加这一行才能画出来 3
# # # filling the area connected
BMred_fill=image_fill(BMred_mask)
BMblue_fill=image_fill(BMblue_mask)
# #--------------------------------------------------------------------
# pyplot.imshow(BMred_fill)
# pyplot.show() #要加这一行才能画出来 4
# # V通道
# pyplot.imshow(BMblue_fill)
# pyplot.show() #要加这一行才能画出来 5

cont_Saver=cnts_find(BMblue_fill,BMred_fill)
print ("Total Contours Found: ",len(cont_Saver)) #cont_Saver.append([cv2.contourArea(d),x, y, w, h])   cv2.contourArea(d)>700:
if len(cont_Saver)>0:
    cont_Saver=np.array(cont_Saver)

    cont_Saver=cont_Saver[cont_Saver[:,0].argsort()].astype(int)
    for conta in range(len(cont_Saver)):
        cont_area,x, y, w, h=cont_Saver[len(cont_Saver)-conta-1]

        #getting the boundry of rectangle around the contours.

        image_found=img[y:y+h,x:x+w]

        crop_image=image_found.copy()
        img0=cv2.cvtColor(image_found, cv2.COLOR_RGB2GRAY)
        img0 = cv2.medianBlur(img0,3)

        crop_image0=cv2.resize(img0, (64, 64))

        # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
        # need to calculate.
        ret,crop_image0 = cv2.threshold(crop_image0,127,255,cv2.THRESH_BINARY)
        descriptor,imagehog  = hog(crop_image0, orientations=8,pixels_per_cell=(4,4),visualize=True)


        # descriptor,imagehog = hog(crop_image0, orientations=8, visualize=True)
        descriptor_pca=pca.transform(descriptor.reshape(1,-1))
        # pyplot.imshow(descriptor)
        # pyplot.show() #要加这一行才能画出来 TypeError: Invalid shape (14112,) for image data
        # cv2.imshow('test', descriptor)
        # cv2.waitKey(0)
        # cv2.destroyWindow('test')
        # class predition of image using SVM
        Predicted_Class=classifier.predict(descriptor_pca)[0]
        print(Predicted_Class)

        if Predicted_Class !=38:
            print ('Predicted Class: ',Predicted_Class)
            ground_truth_image=cv2.imread('F:/TSR/TSRcode/TSR--HSV-SVM--BelgiumTSC-master/TSR--HSV-SVM--BelgiumTSC-master/code/classes_images/'+str(Predicted_Class)+'.png')
            ground_truth_image= ground_truth_image.replace('\\', '/')
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)#drawing a green rectange around it.
            #Putting text on the upward of bounding box
            cv2.putText(img, 'Class: '+str(Predicted_Class), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 6)

            #loading the ground truth class respective to the predicted class
            # displaying the ground truth image
            #ground truth image resize and match according to the sign detected

            try:

                ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                #replaceing the sign image adjacent(left side) to detected sign
                img[y:y+ground_truth_image_resized.shape[0], x-w:x-w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized
            #if sign detected on left boundry then there will be an error because n0 place for image to place then this program run place the image one right side. 
            except:
                #ground truth image resize and match according to the sign detected
                ground_truth_image_resized=cv2.resize(ground_truth_image, (w,h))
                #replaceing the sign image adjacent(right side) to detected sign
                img[y:y+ground_truth_image_resized.shape[0], x+w:x+w+ground_truth_image_resized.shape[1]] = ground_truth_image_resized

            print ('Saving Image as Final_Ouput.png')
            cv2.imwrite('F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\FinalImage\Final_Ouput.png',img)
        else:
            print ('Saving Image as Final_Ouput.png')
            cv2.imwrite('F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\FinalImage\Final_Ouput.png',img)
            


