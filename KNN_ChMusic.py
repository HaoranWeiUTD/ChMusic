# -*- coding: utf-8 -*-
"""
Created on July 04, 2021
by Haoran Wei
This code is used for music intruments recognition, KNN Model
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
import pdb

def spectral_centroid(x, samplerate=44100):
    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
    length = len(x)
    freqs = np.abs(np.fft.fftfreq(length, 1.0/samplerate)[:length//2+1]) # positive frequencies
    return np.sum(magnitudes*freqs) / np.sum(magnitudes) # return weighted mean


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

    # KNN 
    KNN_model = KNeighborsClassifier(n_neighbors=15)
    # train KNN_model
    X = []
    Y = []
    for i in traindata:
        for j in range(len(i[0][0,:])):
            X.append(i[0][:,j])
            Y.append(int(i[1]))
    KNN_model.fit(X, Y)
    
    # test KNN_model
    correct = 0
    all_clips = 0
    predict = []
    ground = []
    for i in testdata:
        X_test = []
        all_clips += 1
        for j in range(len(i[0][0,:])):
            X_test.append(i[0][:,j])
        clip_result = KNN_model.predict(X_test)
        c = Counter(clip_result)
        value, count = c.most_common()[0]
        predict.append(value)
        ground.append(int(i[1]))
        if value == int(i[1]):
            correct += 1
    accuracy = correct/all_clips
    print(accuracy)

    cm  = confusion_matrix(ground, predict, labels=KNN_model.classes_)
    fig, ax = plot_confusion_matrix(conf_mat=cm, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    #plt.title('Confusion Matrix', fontsize=18)
    plt.imshow(cm)
    plt.savefig('KNN_ConfusionMatrix.pdf')
        

