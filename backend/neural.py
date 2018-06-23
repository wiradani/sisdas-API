import csv
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import cross_validate
import itertools
import numpy as np
from sklearn import svm

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


    trainx = train.drop(['Class'], axis=1)
    trainy = train.drop(['b','g','r','h','s','v','pk','berat'], axis=1)


    testx = test.drop(['Class','pk','berat'], axis=1)
    testy = test.drop(['b','g','r','h','s','v','pk','berat'], axis=1)

    data = dataframe.drop(['Class','pk','berat'], axis=1)
    kelas = dataframe.drop(['b','g','r','h','s','v','pk','berat'], axis=1)

    xdata = dataframe.drop(['Class'], axis=1)
    ytarget = dataframe.drop(['b','g','r','h','s','v','pk','berat'], axis=1)

    X=data
    Y=numpy.ravel(kelas)
 
 
    #MLP CLassification backprop
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
          beta_1=0.9, beta_2=0.999, early_stopping=False,
          epsilon=1e-08, hidden_layer_sizes=(10, 7), learning_rate='adaptive',
          learning_rate_init=0.001, max_iter=10000, momentum=0.9,
          nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
          solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
          warm_start=False)
    
    clf.fit(data,kelas)
    
    #Cross validation check
    scores = cross_val_score(clf, data, kelas, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    #Ubah ke persen
    akurasi = scores.mean()*100
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
    
    #Confusion matrix
    class_names = [1,2,3,4,5]
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(xdata, ytarget, random_state=0)

    # Run classifier, using a model that is too regularized (C too low) to see
    # the impact on the results
    classifier = clf
    y_pred = classifier.fit(X_train, y_train).predict(X_test)


    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                          title='Confusion matrix')

    #plt.show()

    return akurasi,kelas




