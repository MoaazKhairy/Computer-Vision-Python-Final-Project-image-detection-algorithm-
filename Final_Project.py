# Code for the final project Computer Vision Task
# Authors @ Abdelrahman Ahmed Ramzy, Ahmed Fawzi Hosni, Moaz Khairy Hussien

import sys
import numpy as np
import pandas as pd
import os
import argparse
import time
from random import randint
# PyQt5
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import pyqtSlot, QSize, QRect
from PyQt5.QtWidgets import QApplication, QDialog, QWidget, QLabel, QMessageBox, QMainWindow, QFileDialog, QComboBox, \
    QRadioButton, QLineEdit
from PyQt5.uic import loadUi
from PyQt5.QtGui import QIcon, QPixmap, QMouseEvent, QPainter
from PyQt5.QtCore import Qt
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import pyqtgraph as pg
# Image Processing
import cv2
import argparse
from skimage import color
from skimage.transform import resize
# Scipy
from scipy import signal
from scipy import misc
import scipy.fftpack as fp
# Math
import math
from math import sqrt, atan2, pi, cos, sin
from collections import defaultdict
from skimage import img_as_float
from skimage.filters import sobel, gaussian
#from skimage.draw import circle_perimeter
from scipy import signal
from scipy import misc
from skimage import color
from scipy.interpolate import RectBivariateSpline


# organize imports
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
import glob
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer
# References:





def OpenedFile(fileName):
    i = len(fileName) - 1
    j = -1
    x = 1

    while x == 1:
        if fileName[i] != '/':
            j += 1
            i -= 1
        else:
            x = 0
    File_Names = np.zeros(j + 1)

    # Convert from Float to a list of strings
    File_Name = ["%.2f" % number for number in File_Names]
    for k in range(0, j + 1):
        File_Name[k] = fileName[len(fileName) - 1 + k - j]  # List of Strings
    # Convert list of strings to a string
    FileName = ''.join(File_Name)  # String
    return FileName

def image_parameters(imagergb):
    imagegray = color.rgb2gray(imagergb)  # np.dot(image[..., :3], [0.299, 0.587, 0.114])
    max = np.max(imagegray)
    min = np.min(imagegray)
    image_size = np.shape(imagegray)
    return imagegray, max, min, image_size

global imageGRAY, imageRGB, Max, Min, imageSize, Clicked, clicker, imagegray, imagergb, imageSource

Clicked = 0


class CV(QMainWindow):
    def __init__(self):
        super(CV, self).__init__()
        loadUi('mainwindow.ui', self)
        self.uploadImage_pushButton.clicked.connect(self.load_image)
        self.start_pushButton.clicked.connect(self.start)
        self.initialization()

    def browser(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        image_source = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        return image_source

    def load_image(self):
        global imageGRAY, imageRGB, Max, Min, imageSize, Clicked, imageSource
        imageSource = self.browser()
        # To make sure the application doesn't crash if no image is loaded
        if imageSource:
            imageRGB = cv2.imread(imageSource)
            imageGRAY, Max, Min, imageSize = image_parameters(imageRGB)
            name = '(' + str(imageSize[0]) + 'X' + str(imageSize[1]) + ')'
            self.label_13.setText(OpenedFile(imageSource))
            self.label_14.setText(name)
            plt.imsave("input.png", cv2.cvtColor(imageRGB, cv2.COLOR_BGR2RGB))
            img = "input.png"
            self.inputImage.setPixmap(QPixmap(img).scaled(self.inputImage.width(), self.inputImage.height()))
            Clicked = 1

    def initialization(self):
        # fixed-sizes for image
        fixed_size = tuple((500, 500))
        
        # path to training data
        train_path = "dataset/train2/"
        
        # bins for histogram
        bins = 8
        
        # feature-descriptor-1: Hu Moments
        def fd_hu_moments(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = cv2.HuMoments(cv2.moments(image)).flatten()
            return feature
        
        # feature-descriptor-2: Haralick Texture
        def fd_haralick(image):
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # compute the haralick texture feature vector
            haralick = mahotas.features.haralick(gray).mean(axis=0)
            # return the result
            return haralick
        
        # feature-descriptor-3: Color Histogram
        def fd_histogram(image, mask=None):
            # convert the image to HSV color-space
            imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # compute the color histogram
            hist_HSV  = cv2.calcHist([imageHSV], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            # normalize the histogram
            cv2.normalize(hist_HSV, hist_HSV)
            histogram_HSV = hist_HSV.flatten()
           
            # convert the image to LAB color-space
            imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # compute the color histogram
            hist_LAB = cv2.calcHist([imageLAB],[0, 1, 2],None,[bins, bins, bins],[0, 256, 0, 256, 0, 256])
            # normalize the histogram
            cv2.normalize(hist_LAB, hist_LAB)
            histogram_LAB = hist_LAB.flatten()
             
            # return the histogram
            return histogram_HSV , histogram_LAB
        
        # get the training labels
        Train_Labels = os.listdir(train_path)
        
        # sort the training labels
        Train_Labels.sort()
        
        # empty lists to hold feature vectors and labels
        global_features = []
        labels = []
        
        # num of images per class
        images_per_class_train = 22
        
        # loop over the training data sub-folders
        for training_name in Train_Labels:
            # join the training data path and each species training folder
            dir = os.path.join(train_path, training_name)
            
            # get the current training label
            current_label = training_name
            
            # loop over the images in each sub-folder
            for x in range(1, images_per_class_train):
                # get the image file name
                file = dir + "/" + str(x) + ".jpg"
                # read the image and resize it to a fixed-size
                image = cv2.imread(file)
                image = cv2.resize(image, fixed_size)    
                # Global Feature extraction
                fv_hu_moments = fd_hu_moments(image)
                fv_haralick   = fd_haralick(image)
                fv_histogram_HSV , fv_histogram_LAB  = fd_histogram(image)
        
                # Concatenate global features
                global_feature = np.hstack([fv_hu_moments, fv_haralick, fv_histogram_HSV, fv_histogram_LAB])
                # update the list of labels and feature vectors
                labels.append(current_label)
                global_features.append(global_feature)
        
        # encode the target labels
        targetNames = np.unique(labels)
        le = LabelEncoder()
        target = le.fit_transform(labels)
        
        # normalize the feature vector in the range (0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = scaler.fit_transform(global_features)
        
        # save the feature vector using HDF5
        h5f_data = h5py.File('output/data.h5', 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
        
        h5f_label = h5py.File('output/labels.h5', 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))
        
        h5f_data.close()
        h5f_label.close()
        
        
        # empty lists to hold feature vectors and labels
        global_features = []
        labels = []
        
        # num of images per class
        images_per_class_test = 25
        
        # loop over the testing data sub-folders
        for training_name in Train_Labels:
            # join the testing data path and each species testing folder
            dir = os.path.join(train_path, training_name)
            
            # get the current training label
            current_label = training_name
            
            # loop over the images in each sub-folder
            for x in range(images_per_class_train, images_per_class_test+1):
                # get the image file name
                file = dir + "/" + str(x) + ".jpg"
                # read the image and resize it to a fixed-size
                image = cv2.imread(file)
                image = cv2.resize(image, fixed_size)    
                # Global Feature extraction
                fv_hu_moments = fd_hu_moments(image)
                fv_haralick   = fd_haralick(image)
                fv_histogram_HSV , fv_histogram_LAB  = fd_histogram(image)
        
                # Concatenate global features
                global_feature = np.hstack([fv_hu_moments, fv_haralick, fv_histogram_HSV, fv_histogram_LAB])
                # update the list of labels and feature vectors
                labels.append(current_label)
                global_features.append(global_feature)
        
        # encode the target labels
        targetNames = np.unique(labels)
        le = LabelEncoder()
        target = le.fit_transform(labels)
        
        # normalize the feature vector in the range (0-1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        rescaled_features = scaler.fit_transform(global_features)
        
        # save the feature vector using HDF5
        h5f_data = h5py.File('output/test_data.h5', 'w')
        h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
        
        h5f_label = h5py.File('output/test_labels.h5', 'w')
        h5f_label.create_dataset('dataset_1', data=np.array(target))
        
        h5f_data.close()
        h5f_label.close()

    def start(self):
        global imageGRAY, imageRGB, Max, Min, imageSize, Clicked, imageSource
        # no.of.trees for Random Forests
        num_trees = 100
        # train_test_split size
        test_size = 0.10
        
        # seed for reproducing same results
        seed = 9
        # path to training data
        train_path = "dataset/train2/"
        # fixed-sizes for image
        fixed_size = tuple((500, 500))
        # bins for histogram
        bins = 8
        # feature-descriptor-1: Hu Moments
        def fd_hu_moments(image):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            feature = cv2.HuMoments(cv2.moments(image)).flatten()
            return feature
        
        # feature-descriptor-2: Haralick Texture
        def fd_haralick(image):
            # convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # compute the haralick texture feature vector
            haralick = mahotas.features.haralick(gray).mean(axis=0)
            # return the result
            return haralick
        
        # feature-descriptor-3: Color Histogram
        def fd_histogram(image, mask=None):
            # convert the image to HSV color-space
            imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # compute the color histogram
            hist_HSV  = cv2.calcHist([imageHSV], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
            # normalize the histogram
            cv2.normalize(hist_HSV, hist_HSV)
            histogram_HSV = hist_HSV.flatten()
           
            # convert the image to LAB color-space
            imageLAB = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            # compute the color histogram
            hist_LAB = cv2.calcHist([imageLAB],[0, 1, 2],None,[bins, bins, bins],[0, 256, 0, 256, 0, 256])
            # normalize the histogram
            cv2.normalize(hist_LAB, hist_LAB)
            histogram_LAB = hist_LAB.flatten()
             
            # return the histogram
            return histogram_HSV , histogram_LAB
        
        # get the training labels
        Train_Labels = os.listdir(train_path)
        
        # import the feature vector and trained labels
        h5f_data = h5py.File('output/data.h5', 'r')
        h5f_label = h5py.File('output/labels.h5', 'r')
        
        global_features_string = h5f_data['dataset_1']
        global_labels_string = h5f_label['dataset_1']
        
        train_features = np.array(global_features_string)
        train_labels = np.array(global_labels_string)
        
        h5f_data.close()
        h5f_label.close()
        
        # import the feature vector and test labels
        h5f_data = h5py.File('output/test_data.h5', 'r')
        h5f_label = h5py.File('output/test_labels.h5', 'r')
        
        test_features_string = h5f_data['dataset_1']
        test_labels_string = h5f_label['dataset_1']
        
        test_features = np.array(test_features_string)
        test_labels = np.array(test_labels_string)
        
        h5f_data.close()
        h5f_label.close()
        
        def perf_measure(clf,train_features, train_labels):
    
            clf.fit(train_features, train_labels)
            prediction = clf.predict(test_features)
            
            accuracy = accuracy_score(test_labels,prediction)
           
            lb = LabelBinarizer()
            lb.fit(test_labels)
            truth = lb.transform(test_labels)
            pred = lb.transform(prediction)
            auc = roc_auc_score(truth, pred, average="macro")
            return (accuracy,auc)
        
        model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
        (accuracy,auc) = perf_measure (model,train_features, train_labels)

        # read the image
        #imageSource =  imageRGB
        image = imageRGB
        
        # resize the image
        image = cv2.resize(image, fixed_size)
        
        # Global Feature extraction
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram_HSV ,fv_histogram_LAB = fd_histogram(image)
        
        # Concatenate global features
        global_feature = np.hstack([fv_hu_moments, fv_haralick, fv_histogram_HSV, fv_histogram_LAB])
        
        # predict label of test image
        prediction = model.predict(global_feature.reshape(1,-1))[0]
        
        # show predicted label on image
        cv2.putText(image, Train_Labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        
        fig = plt.figure()

        # display the output image
        plt.imsave("result.png", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        img = "result.png"
        self.expectedType_label.setText(str(Train_Labels[prediction]))
        self.accuracy_label.setText(str(accuracy))
        self.auc_label.setText(str(auc))

        self.onputImage.setPixmap(QPixmap(img).scaled(self.onputImage.width(), self.onputImage.height()))


if __name__ == "__main__":
    app = 0  # This is the solution As the Kernel died every time I restarted the consol
    app = QApplication(sys.argv)
    widget = CV()
    widget.show()
    sys.exit(app.exec_())