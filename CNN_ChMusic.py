# -*- coding: utf-8 -*-
"""
Created on July 04, 2021
by Haoran Wei
This code is used for music intruments recognition, CNN Model
"""
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import numpy as np 
from pydub import AudioSegment
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from mlxtend.plotting import plot_confusion_matrix
import soundfile as sf
import librosa
import os
import pickle
import pdb
import tensorflow as tf
import scipy.io 
import scipy.signal
from skimage.transform import resize
import cv2


if __name__ == "__main__":
    music_path = "G:\DataSet\Music\ChMusic\Musics"
    # make decision every 5 mimutes
    target_clip_length = 5 
    print("Let's start now")
    print(music_path)

    filter=[".wav"]
    music_list = []
    # go troughou this folder to find out related wav files
    for maindir, subdir, file_name_list in os.walk(music_path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]  
            if ext in filter:
                # if this file using .wav format
                # the path of this wav file will be added to musci_list
                music_list.append(apath)
    
    # seperate traindata and testdata from music_list
    traindata = []
    testdata = []
    for i in music_list:   
        music = AudioSegment.from_wav(i)
        samplerate = music.frame_rate
        # int will find integer less than (music.duration_seconds//target_clip_length)
        clip_number = int(music.duration_seconds//target_clip_length)
        for k in range(clip_number):
            music[k*target_clip_length*1000:(k+1)*target_clip_length*1000].export("./tem_clip.wav", format="wav")
            x, sr = librosa.load("./tem_clip.wav")
            # hop_length 512, 512/22050 -> 23.22, 23.22*216 ->5.01552
            mfcc_tem = librosa.feature.mfcc(x, sr, n_mfcc=20)
            if i[-5] == '5':
                strlist = i.split('\\')
                strlist[-1][:strlist[-1].find(".")]
                testdata.append([mfcc_tem,strlist[-1][:strlist[-1].find(".")]])
            else:
                strlist = i.split('\\')
                strlist[-1][:strlist[-1].find(".")]
                #  mfcc_tem.shape -> (20, 216), label -> (1)
                traindata.append([mfcc_tem,strlist[-1][:strlist[-1].find(".")]])

    # train KNN_model
    train_X = []
    train_Y = []
    test_X = []
    test_Y = []
    for i in traindata:
        train_X.append(i[0])
        train_Y.append(int(i[1])-1)
    for i in testdata:
        test_X.append(i[0])
        test_Y.append(int(i[1])-1)
    train_X = np.array(train_X)
    train_X = np.expand_dims(train_X, axis=3)
    train_Y = np.array(train_Y)
    test_X = np.array(test_X)
    test_X = np.expand_dims(test_X, axis=3)
    test_Y = np.array(test_Y)

    # Define model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=3, activation=tf.nn.relu,padding="same",input_shape=(20,216,1)),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)), 
    tf.keras.layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu,padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),   
    tf.keras.layers.Conv2D(128, kernel_size=3, activation=tf.nn.relu,padding="same"),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)),   
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(11, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    pdb.set_trace()
    model.fit(train_X, train_Y, epochs=15,batch_size=32)
    loss, accuracy = model.evaluate(test_X, test_Y, batch_size=1)
    predict = model.predict(test_X)
    tf.keras.models.save_model(model,'myModel_CNN.h5',overwrite=True,include_optimizer=True) 

        







        
        
        

            



    
    

