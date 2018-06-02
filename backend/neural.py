import csv
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import sklearn

class Neural:
  def klasifikasi(self):
    #Read Data
    os.chdir('/home/adhan/Projek/sisdas-API/backend/temp')
    filename = open('datatomat.csv', 'r')
    names = ['b', 'g', 'r', 'h', 's', 'v', 'Class','pk','berat']
    dataframe = pandas.read_csv(filename, names=names)

    namafile = open('predict.csv', 'r')
    atribut = ['b', 'g', 'r', 'h', 's', 'v','to']
    dataPredict = pandas.read_csv(namafile, names=atribut)
    dataPredict = dataPredict.drop('to', axis=1)


    #Split Data
    train, test = train_test_split(dataframe, test_size=0.8)


    trainx = train.drop('Class', axis=1)
    trainy = train.drop(['b','g','r','h','s','v'], axis=1)


    testx = test.drop('Class', axis=1)
    testy = test.drop(['b','g','r','h','s','v'], axis=1)

    data = dataframe.drop(['Class','pk','berat'], axis=1)
    kelas = dataframe.drop(['b','g','r','h','s','v','pk','berat'], axis=1)


    #MLP CLassification
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
          beta_1=0.9, beta_2=0.999, early_stopping=False,
          epsilon=1e-08, hidden_layer_sizes=(10, 7), learning_rate='adaptive',
          learning_rate_init=0.001, max_iter=1000000, momentum=0.9,
          nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
          solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
          warm_start=False)
    
    clf.fit(data,kelas)
    
    akurasi = clf.score(data,kelas)*100
    akurasi=int(akurasi)
    akurasi = str(akurasi)+" " + "%"
    kelas=clf.predict(dataPredict)
    kelas=int(kelas)
    if kelas == 1:
      kelas = '[1] mentah'
    elif kelas == 2:
      kelas = '[2] setengah matang'
    elif kelas == 3:
      kelas = '[3] cukup matang'
    elif kelas == 4:
      kelas = '[4] matang'
    else:
      kelas = '[5] sangat matang'
  

    return akurasi,kelas




