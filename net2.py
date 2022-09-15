# LSTM for international airline passengers problem with regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
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
dataset = dataframe.values[1:1001]
dataset = dataset.astype('float32')
#repeat with other data set
dataframe2 = read_csv('dat2.csv', usecols=[1], engine='python')
dataset2 = dataframe2.values[1:1001]
dataset2 = dataset2.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
#for dat2
dataset2 =  scaler.fit_transform(dataset2)
# split into train and test sets
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
#only test for dataset2
test2 = dataset2
# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
#create test2
test2, test2Y = create_dataset(test2, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
test2 = numpy.reshape(test2, (test2.shape[0], 1, test2.shape[1]))
# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, look_back)))
#model.add(Dense(10, activation='relu'))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=64, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
farPredict = model.predict(test2)
# print(trainPredict[0])
#this section was an attempt to correct the problem with too much fitting
# tp = []
# for p in trainPredict:
#     tp.append(p[0])
# trainPredict = tp
# tp = []
# for p in testPredict:
#     tp.append(p[0])
# testPredict = tp
# invert predictions
farPredict = scaler.inverse_transform(farPredict)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
fig, axs = plt.subplots(2)
fig.suptitle('Time series data vs forecast')
axs[0].plot(scaler.inverse_transform(dataset))
#axs[0].plot(trainPredictPlot)
axs[0].plot(testPredictPlot)
axs[1].plot(scaler.inverse_transform(dataset2))
axs[1].plot(farPredict)
plt.show()
