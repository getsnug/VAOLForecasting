import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import numpy as np
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)
# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset
dataframe = read_csv('dat.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset[1:1000]
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.9)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1
look_back = 6
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(Dense(10, input_shape=(1, look_back), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=1, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
curr = trainX[-1]
currList = []
for c in curr:
    currList.append([c])
testPredict = []
for i in range(len(testX)):
    print(len(testX))
    currList = np.array(currList)
    preDat = currList
    pre = model.predict(currList)
    testPredict.append(pre[0])
    print(currList[0])
    cList = []
    for c in currList[0][0]:
        cList.append(c)
    cList = [[cList[1:]]]
    cList[0][0].append(pre[0][0][0])
    currList = cList

#testPredict = model.predict(testX)
# invert predictions
print(trainPredict)
print(testPredict)
tP = []
for p in trainPredict:
    tP.append(p[0])
trainPredict = tP
tP = []
for p in testPredict:
    tP.append(p[0])
testPredict = tP
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
#print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
# with open('/output.csv', 'w') as f:
#     # create the csv writer
#     writer = csv.writer(f)
#     # write a row to the csv file
#     row = []
#     for i in range(len(testX)):
#         row = [testX[i]]
#         row.append(trainPredict[i])
#         writer.writerow(row)
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
