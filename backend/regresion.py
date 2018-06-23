import csv
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score


class Regresion:
  def reges(self):
    #Read Data
    os.chdir('/home/adhan/Projek/sisdas-API/backend/temp')
    filename = open('datatomat.csv', 'r')
    names = ['b', 'g', 'r', 'h', 's', 'v', 'Class','pk','berat']
    dataframe = pandas.read_csv(filename, names=names)

    namafile = open('predict.csv', 'r')
    atribut = ['b', 'g', 'r', 'h', 's', 'v','to']
    dataPredict = pandas.read_csv(namafile, names=atribut)
    dataPredict = dataPredict.drop(['b','g','r','h','s','v'], axis=1)


    #Split Data
    data = dataframe.drop(['b','g','r','h','s','v','Class','berat'], axis=1)
    kelas = dataframe.drop(['b','g','r','h','s','v','Class','pk'], axis=1)

    kelas=numpy.ravel(kelas)

    i=5
    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, data, kelas, cv=i)

    y=kelas

    #Cross validation check
    scores = cross_val_score(lr, data, kelas, cv=i)
    print("Accuracy regres: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #PLot
    fig, ax = plt.subplots()
    ax.scatter(y, predicted, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    #jika ingin menampilkan hasil plot
    #plt.show()

    reg = linear_model.LinearRegression()
    reg.fit (data,kelas)

    #cek mean square error
    mse=mean_squared_error(data,kelas)
    
    score = scores.mean()*100
    berat = float(reg.predict(dataPredict)-8)
    berat = format(round(berat,2))
    score=format(round(score,1))
    str(berat)
    berat = berat +" "+"gram"
    score = str(score) +" "+"%"
    mse=format(round(mse,2))
   

    return berat,score,mse
