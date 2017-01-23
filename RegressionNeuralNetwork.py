import csv
import numpy
import time
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import model_from_json

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

data, targets = readData()
x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3)

inp_dim = int((data.shape)[1])
layers = [inp_dim, 180, 230, 150, 70, 20, 1]
mod_name = '-'.join([str(x) for x in layers])
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(layers[0], input_dim=layers[0], init='normal', activation='relu'))
	for layer in layers[1:-1]:
          model.add(Dense(layer, init='normal', activation='relu'))
	model.add(Dense(layers[-1], init='normal'))
	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.compile(loss='mse', optimizer='adam')
	return model

# evaluate model with standardized dataset
model = baseline_model()
t0 = time.clock()
print ("Training. . . . . . . .")
model.fit(x_train, y_train, nb_epoch=100)
print ("Time taken:", time.clock()-t0, "seconds")

t0 = time.clock()
print ("Saving model ...... ")
# serialize model to JSON
model_json = model.to_json()
with open("../models/model-"+mod_name+".json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("../models/model-"+mod_name+".h5")
print ("Saved model to disk")
print ("Time taken:", time.clock()-t0, "seconds")

'''json_file = open("../models/model-"+mod_name+".json", 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("../models/model-"+mod_name+".h5")
model.compile(loss='mean_squared_error', optimizer='adagrad')
print("Loaded model from disk")'''

t0 = time.clock()
print ("Testing. . . . . . . .")
scores = model.evaluate(x_test, y_test)
print(scores)
print ("Time taken:", time.clock()-t0, "seconds")


testingdata, ids = readData(train=False, test=True)
predictions = list(model.predict(testingdata))
with open("../results/ResultFile-"+mod_name+".csv", 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'loss'])
    for x,y in zip(ids, predictions):
        writer.writerow([x, y[0]])
        

