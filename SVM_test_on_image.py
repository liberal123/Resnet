import glob
import pandas as pd  # as means that we use pandas library short form  as pd
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt # matplotlib is big library, we are just calling pyplot function 
                                    # for showing images
from skimage.feature import hog #We are calling only hog  

from sklearn.decomposition import PCA # Calling the PCA funtion from sklearn
from sklearn.svm import SVC # # Calling the SVM function from sklearn
from sklearn.externals import joblib # Calling the joblib function from sklearn, use for model saving 
                                     # and loading.
import sys                                     
# Loading the mode into same name
# pca = joblib.load('pca.pkl')
# classifier = joblib.load('svm.pkl')
pca = torch.load('F:\python project\交通标识识别项目1\model\pca.pkl')
classifier = torch.load('F:\python project\交通标识识别项目1\model\svm.pkl')


from skimage.exposure import exposure #for displaying th hog image.
# img_path=csv_files_Testing[main_Testing['ClassId'][image_number]].split('GT')[0]+main_Testing['Filename'][image_number]
# img_path=sys.argv[1]
#img_path="dataset/Testing/00014/00389_00001.ppm

#
## Initilize the 3 axis so that we can plot side by side
#fig, (ax1, ax2,ax3) = plt.subplots(1, 3, figsize=(10, 10), sharex=True, sharey=True)
#
##ploting crop image
#ax1.axis('off')
#ax1.imshow(cv2.cvtColor(crop_image,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
#ax1.set_title('Crop image')
#
## Rescale histogram for better display,Return image after stretching or shrinking its intensity levels
#hog_image_rescaled = exposure.rescale_intensity(imagehog, in_range=(0, 10))
##ploting Hog image
#ax2.axis('off')
#ax2.imshow(imagehog, cmap=plt.cm.gray)
#ax2.set_title('Histogram of Oriented Gradients')
##ploting Orignal image
#ax3.axis('off')
#ax3.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB), cmap=plt.cm.gray)
#ax3.set_title('Orignal Image')
#plt.savefig('Result.png')
# class predition of image using SVM
Test_Images_Directory=r'F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Testing'
csv_files_testing=glob.glob(Test_Images_Directory+'/**/*.csv',recursive=True) # recursive true means it will check
                                                                          # folder inside the folder as well.
print(csv_files_testing) #这里没错
main_Testing=pd.read_csv(csv_files_testing[0],sep=';') # reading first csv and assining it main csv.
print(main_Testing)
# print(type(main_Training))
for i in range(1,len(csv_files_testing)): # for loop iteration from 1 to number of files, by this way can get the all the files read by csv_files using glob.
    new_doc=pd.read_csv(csv_files_testing[i],sep=';') # reading the new csv file as new doc
    # main_Training=main_Training.append(new_doc, ignore_index=True) # appending the csv files making a big csv that consists of all the csv files.
    # main_Training=pd.merge(main_Training,new_doc,on='Filename')
    frame=[main_Testing,new_doc]
    main_Testing = pd.concat(frame)
print ('Total Images found in ',csv_files_testing,' :',len(main_Testing))
# print(main_Training)
# accuracy=0
def PredictAccuracy(main_Testing,Test_Images_Directory):
    TruePredicted = 0
    for i in range(0, len(main_Testing)):  # len(main)
        # we are getting the image path from csv_files name thats is dataset/Training\\00000\\GT-00000.csv
        # then spliting it at GT and we have dataset/Training\\00000\\ after that adding the name of one
        # example that we are dealing like dataset/Testing\\00001\\01983_00002.ppm
        img_path = Test_Images_Directory + '/00000'[:-len(str(main_Testing['ClassId'].values[i]))] + str(
            main_Testing['ClassId'].values[i]) + '/' + main_Testing['Filename'].values[i]
        # -------------------------重要提示------------------------------------
        # 这里改了好久好久 应该放在另外的文件直接测试会很快 这里main['ClassId'][i]改成main['ClassId'].values[i]就可以
        # main['Filename'][i]改成main['Filename'].values[i]就可以
        # 应该是之前把每个文件夹合并为main_Training的时候把格式改掉了--append改成了concat 所以读的方式要改变
        img_path = img_path.replace('\\', '/')
        # print(Images_Directory) #F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Training
        # print(img_path)
        # print('12345'[:-len(str(main['ClassId'][i]))])#是错的
        img = cv2.imread(img_path)
        crop_image = img
        img0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img0 = cv2.medianBlur(img0, 3)

        crop_image0 = img0
        crop_image0 = cv2.resize(crop_image0, (64, 64))

        # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
        # need to calculate.
        ret, crop_image0 = cv2.threshold(crop_image0, 127, 255, cv2.THRESH_BINARY)
        descriptor, imagehog = hog(crop_image0, orientations=8, pixels_per_cell=(4, 4), visualize=True)

        # descriptor,imagehog = hog(crop_image0, orientations=8, visualize=True)
        descriptor_pca = pca.transform(descriptor.reshape(1, -1))
        Predicted_Class = classifier.predict(descriptor_pca)[0]
        print('Predicted Class: ', Predicted_Class)
        # print(type(Predicted_Class))#<class 'numpy.int64'>
        if Predicted_Class==main_Testing['ClassId'].values[i]:
            TruePredicted=TruePredicted+1
    return TruePredicted
# print(type(main_Testing['ClassId'].values[i])) #<class 'numpy.int64'>
TruePredicted=PredictAccuracy(main_Testing,Test_Images_Directory)
acc=TruePredicted/len(main_Testing)
print(acc)