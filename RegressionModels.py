import csv
import numpy
import time
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn import ensemble
from sklearn import svm

seed = 7
numpy.random.seed(seed)
lben = preprocessing.LabelEncoder()

def readData(train=True, test=False):
  if train:
    ## READ AND SEPARATE DATA ##
    features = []
    loss = []
    t0 = time.clock()
    print ("Reading data from file........")
    with open('../data/train.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features.append(row[1:-1])
            loss.append(row[-1])
        f.close()
    print ("Time taken:", time.clock()-t0, "seconds")
    t0 = time.clock()
    print ("Processing categorical data........")
    npfeatures = numpy.array(features[1:])
    loss = numpy.array(loss[1:]).astype(float)
    for i in range(len(features[0])):
        if 'cat' in features[0][i]:
            npfeatures[:, i] = lben.fit_transform(npfeatures[:, i])
    print ((npfeatures.shape)[1], "features")
    npfeatures = npfeatures.astype(float)
    print ("Time taken:", time.clock()-t0, "seconds")
    return npfeatures, loss

  elif test:
    features = []
    ids = []
    t0 = time.clock()
    print ("Reading data from file........")
    with open('../data/test.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            ids.append(row[0])
            features.append(row[1:])
        f.close()
    print ("Time taken:", time.clock()-t0, "seconds")
    ids = ids[1:]
    npfeatures = numpy.array(features[1:])
    for i in range(len(features[0])):
        if 'cat' in features[0][i]:
            npfeatures[:, i] = lben.fit_transform(npfeatures[:, i])
    print ((npfeatures.shape)[1], "features")
    npfeatures = npfeatures.astype(float)
    print ("Time taken:", time.clock()-t0, "seconds")
    return npfeatures, ids
    

################ SCALING AND NORMALIZATION ##########
##npfeatures = preprocessing.scale(npfeatures)
##loss = preprocessing.scale(loss)

data, targets = readData()
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)

regressor = ensemble.AdaBoostRegressor(n_estimators=100, loss='square')
name = 'AdaBoostRegressor'
'''regressors = [
            (linear_model.LinearRegression(), 'LinearRegression'),
            (linear_model.Ridge(alpha=0.5), 'RidgeRegression'),
            (linear_model.RidgeCV(alphas = [0.1, 1.0, 10.0]), 'RidgeCrossValidation'),
            (linear_model.Lasso(alpha=0.1), 'Lasso'),
            (linear_model.LassoCV(alphas = [0.1, 1.0, 10.0]), 'LassoCrossValidation')]
for regressor, name in regressors:
    print (name)
    t0 = time.clock()
    print ("Training. . . . . ")
    regressor.fit(x_train, y_train)
    print ("Time taken:", time.clock()-t0, "seconds")
    t0 = time.clock()
    print ("Testing.. .. .. .. .. ")
    print ("Score:", regressor.score(x_test, y_test))
    print ("Time taken:", time.clock()-t0, "seconds")'''

t0 = time.clock()
print ("Training. . . . . . . .")
regressor.fit(x_train, y_train)
print ("Time taken:", time.clock()-t0, "seconds")

t0 = time.clock()
print ("Testing. . . . . . . .")
scores = regressor.score(x_test, y_test)
print(scores)
print ("Time taken:", time.clock()-t0, "seconds")


testingdata, ids = readData(train=False, test=True)
predictions = list(regressor.predict(testingdata))
with open('../results/ResultFile'+name+'.csv', 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'loss'])
    for x,y in zip(ids, predictions):
        writer.writerow([x, y])
        

