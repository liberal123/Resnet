import glob
import pandas as pd  # as means that we use pandas library short form  as pd
import cv2
import numpy as np
from matplotlib import pyplot as plt # matplotlib is big library, we are just calling pyplot function
                                    # for showing images
from skimage.feature import hog #We are calling only hog

from sklearn.decomposition import PCA # Calling the PCA funtion from sklearn
from sklearn.svm import SVC # # Calling the SVM function from sklearn
from sklearn.externals import joblib
import matplotlib
import skimage
import sklearn
Test_Images_Directory=r'F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Testing'
Training_Images_Directory=r'F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Training'
csv_files_training=glob.glob(Training_Images_Directory+'/**/*.csv',recursive=True) # recursive true means it will check
                                                                          # folder inside the folder as well.
# print(csv_files_training) #这里没错
main_Training=pd.read_csv(csv_files_training[0],sep=';') # reading first csv and assining it main csv.
# print(main_Training)
# print(type(main_Training))
for i in range(1,len(csv_files_training)): # for loop iteration from 1 to number of files, by this way can get the all the files read by csv_files using glob.
    new_doc=pd.read_csv(csv_files_training[i],sep=';') # reading the new csv file as new doc
    # main_Training=main_Training.append(new_doc, ignore_index=True) # appending the csv files making a big csv that consists of all the csv files.
    # main_Training=pd.merge(main_Training,new_doc,on='Filename')
    frame=[main_Training,new_doc]
    main_Training = pd.concat(frame)
print ('Total Images found in ',Training_Images_Directory,' :',len(main_Training))
# print(main_Training)
print(main_Training.values)
sss = []  # making a new list.
oneexample = []
for i in range(len(main_Training.values)):  # iterating 0-len(main_Training.values) that is training dataset lenght.
    if main_Training.values[i, -1] not in sss:
        oneexample.append(
            main_Training.values[i, :])  # appending the main_Training dataset row number i.每个文件夹挑一个图片所有的大小特征保存
        sss.append(main_Training.values[i, -1])  # appending the class id of row number i 、就保存1 35 45这些数字

# Making a new pandas dataframe from oneexample list. and giving it columns names.
One_Example = pd.DataFrame(oneexample,
                           columns=['Filename', 'Width', 'Height', 'Roi.X1', 'Roi.Y1', 'Roi.X2', 'Roi.Y2', 'ClassId'])
print(One_Example)  # 总共8个
print ('Saving the classes Images in "classes_images"')
for i in range(len(One_Example)): # range depends on number of classes
    # we are getting the image path from csv_files name thats is dataset/Training\\00000\\GT-00000.csv
    # then spliting it at GT and we have dataset/Training\\00000\\ after that adding the name of one
    # example that we are dealing like dataset/Testing\\00001\\01983_00002.ppm
    img_path=csv_files_training[i].split('GT')[0]+One_Example['Filename'][i]
    img_path=img_path.replace('\\','/')
    # print(img_path)
    img = cv2.imread(img_path) # opencv use for reading image.
    # cv2.imshow(img)
    # print(img) #------------之前打印出none的原因是路径中有中文 所以说cv2.imread（）里面不能有中文路径----------
    # croping the image based on the coordinates given us by csv files
    crop_image=img[One_Example['Roi.Y1'][i]:One_Example['Roi.X2'][i],One_Example['Roi.X1'][i]:One_Example['Roi.Y2'][i]]
    # print(crop_image)
    cv2.imwrite('F:/TSR/TSRcode/TSR--HSV-SVM--BelgiumTSC-master/TSR--HSV-SVM--BelgiumTSC-master/code/classes_images/'+str(One_Example['ClassId'][i])+'.png',crop_image) # use for saving the image.
    #保存图片成功
def images_to_hog(main, Images_Directory):  # function defining that can be call for both test and training
    Features = []
    Labels = []
    for i in range(0, len(main)):  # len(main)
        # we are getting the image path from csv_files name thats is dataset/Training\\00000\\GT-00000.csv
        # then spliting it at GT and we have dataset/Training\\00000\\ after that adding the name of one
        # example that we are dealing like dataset/Testing\\00001\\01983_00002.ppm
        img_path = Images_Directory + '\\00000'[:-len(str(main['ClassId'][i]))] + str(main['ClassId'][i]) + '\\' + \
                   main['Filename'][i]
        print(
            Images_Directory)  # F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Training
        print(str(img_path))
        print('12345'[:-len(str(main['ClassId'][i]))])
        img = cv2.imread(img_path)  # opencv use for reading image.
        # croping the image based on the coordinates given us by csv files
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 3)
        crop_image = img[main['Roi.Y1'][i]:main['Roi.X2'][i], main['Roi.X1'][i]:main['Roi.Y2'][i]]
        crop_image = cv2.resize(crop_image, (64, 64))  # Resize the image to 64*64.
        # Apply Hog from skimage library it takes image as crop image.Number of orientation bins that gradient
        # need to calculate.
        ret, crop_image = cv2.threshold(crop_image, 127, 255, cv2.THRESH_BINARY)  # 返回的第一个参数为阈值，第二个为结果图像
        descriptor = hog(crop_image, orientations=8, pixels_per_cell=(4, 4))
        Features.append(descriptor)  # hog features saving
        Labels.append(main['ClassId'][i])  # class id saving

    Features = np.array(Features)  # converting to numpy array.
    Labels = np.array(Labels)
    return Features, Labels

#
# Features_Training, Labels_Training = images_to_hog(main_Training,
#                                                    Training_Images_Directory)  # giving values to images_to_hog function
# print('Training HOG output Features shape : ', Features_Training.shape)
# print('Training HOG output Labels shape: ', Labels_Training.shape)
main=main_Training
print(main_Training)
Images_Directory='F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Training'
img_path=Images_Directory+'/00000'[:-len(str(main['ClassId'].values[i]))]+str(main['ClassId'].values[i])+'/'+main['Filename'].values[107]
print(Images_Directory) #F:\TSR\TSRcode\TSR--HSV-SVM--BelgiumTSC-master\TSR--HSV-SVM--BelgiumTSC-master\code\Dataset\Training
print(img_path)
# print(main['Filename'].values[107])
# print('/00000'[:-len(str(main['ClassId'][i]))])
print(main['ClassId'].values[924])
print(main['ClassId'][0])
print(main['Roi.Y1'].values[0])
