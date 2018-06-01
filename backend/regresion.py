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

    reg = linear_model.LinearRegression()
    reg.fit (data,kelas)

    print('Coefficients: \n', reg.coef_)
    mse=mean_squared_error(data,kelas)
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(data,kelas))
    score = float(reg.score(data,kelas)*100)
    berat = float(reg.predict(dataPredict)-8)
    berat = format(round(berat,2))
    score=format(round(score,2))
    str(berat)
    berat = berat +" "+"gram"
    score = str(score) +" "+"%"

    return berat,score,mse
