import csv
import numpy
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer
import pandas
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import os



#Read Data
os.chdir('/home/adhan/Projek/sisdas-API/backend/temp')
filename = open('diabetes.csv', 'r')
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age','Class']
dataframe = pandas.read_csv(filename, names=names)

#Normalize Data
# separate array into input and output components
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
scaler = Normalizer().fit(X)
normalizedData = scaler.transform(X)
# summarize transformed data
numpy.set_printoptions(precision=3)
#print(normalizedX[0:5,:])

#Split Data
train, test = train_test_split(dataframe, test_size=0.2)

trainx = train.drop('Class', axis=1)
trainy = train.drop(train.columns[[0, 1, 2,3,4,5,6,7]], axis=1)

testx = test.drop('Class', axis=1)
testy = test.drop(test.columns[[0, 1, 2,3,4,5,6,7]], axis=1)


#MLP CLassification
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=1)

#train data
clf.fit(trainx, trainy)

#predict data
clf.predict(testx)

#print score
print("Training set score: %f" % clf.score(trainx, trainy))
print("Test set score: %f" % clf.score(testx, testy))
print(pandas.DataFrame(clf.predict_proba(testx), columns=clf.classes_))