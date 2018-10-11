#Import
import quandl
import pandas as pd
from stockstats import StockDataFrame as Sdf
from ta import * 
import datetime
import numpy as np
from termcolor import colored, cprint
import time
import matplotlib.pyplot as plt

#ML Imports
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split


#Create random seed for reproducibility
np.random.seed(7)

# AVX/FMA warning disable
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Pandas dataframe formatting
pd.options.mode.chained_assignment = None  
pd.set_option('display.max_columns', None)

#API Setup
quandl.ApiConfig.api_key = 'rf1_rwN3HiMDyGBBYcef'
quandl.ApiConfig.api_version = 'todaysDate'

#Data import options
yearsAgo = 5
importColumns = ['Ticker', 'Date', 'Adj_Open', 'Adj_Close', 'Adj_High', 'Adj_Low', 'Adj_Volume']
dataColumns = ['Ticker', 'Date', 'Open', 'Close', 'High', 'Low', 'Volume']
indicatorColumns = ['RSI', 'CCI', 'MACD', 'STO']
indicatorColumns2 = ['Date', 'RSI', 'CCI', 'MACD', 'STO']
# holdTimes = [14]
# holdTimesStr = ['14']
holdTimes = [7, 14, 28]
holdTimesStr = ['7', '14', '28']
holdTimesStr2 = ['Date', '7', '14', '28']

#Methods

#Input: Number of years back to get data from
#Output: Todays date and date years back
def getDate(yearsAgo):
	todaysDate = datetime.datetime.today().strftime('%Y-%m-%d')
	timeSet = str(int(todaysDate[:4]) - yearsAgo) + todaysDate[4:]
	print('Todays date: ' + colored(todaysDate, 'cyan'))
	print('Start date: ' + colored(timeSet, 'green'))

	return todaysDate, timeSet

#Input: Stock ticker
#Output: Adjusted data values
def getData(ticker, timeSet, todaysDate):
	df = quandl.get('EOD/' + ticker, start_date=timeSet, end_date=todaysDate, paginate=True)
	df = df.fillna(0)

	df = df.reset_index()

	df['Ticker'] = df['Open']
	for val in df['Ticker']: 
		df['Ticker'].replace(val, ticker.upper(), inplace = True)

	df = df[importColumns]
	df.rename(columns={'Adj_Open':'Open','Adj_High':'High',  'Adj_Low':'Low',  'Adj_Close':'Close', 'Adj_Volume':'Volume'}, inplace=True)
	return df

#Input: Dataframe and name for CSV save file
#Saves data to CSV file
#Output: Dataframe
def saveAndRead(df, csvName):
	df.to_csv('csvfiles/' + csvName + '.csv', sep=',', encoding = 'utf-8', index=False, header=True) 
	rawData = pd.read_csv('csvfiles/' + csvName + '.csv', engine = 'python', sep=",", header=0, dtype={'ticker':str}, error_bad_lines=False)
	df = rawData.copy(deep=True)
	return df

#Input: Dataframe
#Output: Dataframe with indicators from indicatorColomns list
def addIndicatorData(df):
	fillna = False

	df['RSI'] = rsi(df['Close'], n=12, fillna=fillna)
	df['CCI'] = cci(df['High'], df['Low'], df['Close'], n=20, fillna=fillna)
	df['MACD'] = macd(df['Close'], n_fast=12, n_slow=26, fillna=fillna)
	df['MACD Diff'] = macd_diff(df['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)
	df['MACD Signal'] = macd_signal(df['Close'], n_fast=12, n_slow=26, n_sign=9, fillna=fillna)
	df['STO'] = stoch(df['High'], df['Low'], df['Close'], n=14, fillna=fillna)
	df['STO Signal'] = stoch_signal(df['High'], df['Low'], df['Close'], n=14, d_n=3, fillna=fillna)
	df['ADX'] = adx(df['High'], df['Low'], df['Close'], n=14, fillna=fillna)

	finalColumns = dataColumns
	for val in indicatorColumns:
		finalColumns.append(val)

	df = df[finalColumns]

#Removes first 10% of data so indicators ramp up
	size = df['Ticker'].size
	newSize = size - int(size*.1)
	df = df.tail(newSize)
	print(colored('Removed first ' + str(size - newSize) + ' values for indicator stabilization. ' + str(newSize) + ' values remaining.', 'yellow'))
	print('New start date: ' + colored(str(df.iat[0,1]).split()[0], 'green'))
	df = df.reset_index(drop=True)

	return df

#Input: Df with indicators
#Output: Buy strength for various time periods based on close price
def computeBuyStrengths(df, numDays):
	closeValues = df['Close'].tolist()

	for days in numDays:
		delta = []
		for val in range(0, len(closeValues) - (days + 1)):
			delta.append(closeValues[val + days] - closeValues[val])

		ending = [0] * (days + 1)
		for val in ending:
			delta.append(val)

		series = pd.Series(delta)
		df[str(days)] = series.values

	df = df.head(len(closeValues) - (numDays[len(numDays) - 1] + 1))
	print(colored('Removed last ' + str(numDays[len(numDays) - 1] + 1) + ' values for output values.', 'yellow'))

	print('New end date: ' + colored(str(df.iat[df.shape[0] - 1,1]).split()[0], 'green'))
	return df

'''
Formats df into lists for ML

Without reformat: df is full and organized by date
With reformat: Output is large list, each index representing one day, and all indicator
data for that day

Input: df with output columns
Output: List of inputs, list of outputs
'''
def machineFormat(df):
	X = df[indicatorColumns]
	Y = df[holdTimesStr]

	inputs = []
	for val in range(X.shape[0]):
		inputs.append(X.loc[val].tolist())

	outputs = []
	for val in range(Y.shape[0]):
		outputs.append(Y.loc[val].tolist())

	return inputs, outputs

#Input: df
#Output: X and Y with date as index
def reformat(df):
	X = df
	X = df[indicatorColumns2]
	X = X.set_index('Date')

	Y = df
	Y = df[holdTimesStr2]
	Y = Y.set_index('Date')

	return X, Y


#Runs program
def run(indicatorColumns, yearsAgo):
	todaysDate, timeSet = getDate(yearsAgo)
	ticker = input('Please Enter a Ticker: ')
	print('\nImporting data for ' + colored(ticker.upper(), 'cyan')) ##Convert ticker to company name
	df = getData(ticker, timeSet, todaysDate)
	df = addIndicatorData(df)
	df = computeBuyStrengths(df, holdTimes)
	inputs, outputs = machineFormat(df)
	X, Y = reformat(df)
	df = saveAndRead(df, ticker)

	print(colored('Indicators used: ', 'red') + str(indicatorColumns)) 

	return df, X, Y, inputs, outputs


#Output block
df, X, Y, inputs, outputs = run(indicatorColumns, yearsAgo)

'''
ML Notes:
Proposed loss function: MSE
Secondary: binary_crossentropy (have to change outputs to [0,1])

Proposed Last Layer Activation: None/sigmoid
'''
def model(inputs, outputs):
	x_train, x_test, y_train, y_test = train_test_split(
		inputs, outputs, test_size=0.15, random_state=7)

	cprint('------------Training Model------------', 'white', attrs=['bold'])
	start_time = time.time()

	batch_size = 16
	epochs = 10

	model = Sequential()

	model.add(Dense(1000, input_dim=len(indicatorColumns), activation='relu'))
	model.add(Dense(500, activation='relu'))

	model.add(Dropout(.1))

	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))

	model.add(Dense(len(holdTimes)))
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	history = model.fit(np.array(x_train), np.array(y_train), epochs=epochs, batch_size=batch_size, 
		validation_data=(np.array(x_test),np.array(y_test)), shuffle=True, verbose=1)

	end_time = time.time()
	accuracy = history.history['val_acc'][epochs - 1]
	loss = history.history['val_loss'][epochs - 1]
	accuracy2 = str(float(int(accuracy * 10000)) / 100)
	loss2 = str(float(int(loss * 100)) / 100)

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model Accuracy: ' + accuracy2[:5] + ' Batch Size: ' + str(batch_size) + ' Number of Epochs: ' + str(epochs))
	plt.ylabel('Accuracy/Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test', 'Loss'], loc='upper left')
	fig1 = plt.gcf()
	plt.show()
	fig1.savefig("latest.png")

	cprint('\n------------Training Results------------', 'white', attrs=['bold'])
	cprint('Batch size: ' + colored(batch_size, 'cyan') + ' Number of epochs: ' + colored(epochs, 'cyan'))
	print('Accuracy: %s' % (accuracy2))
	print('Loss: %s' % (loss2))
	total_time = end_time - start_time
	a = str(int(total_time / 60))
	b = str(int(total_time % 60))
	print('Time Taken: ' + colored(a, 'cyan') + ' Minutes, ' + colored(b, 'cyan') + ' Seconds')

	return model

def makePrediction(model, X):
	predicting = True
	while(predicting):
		try:
			prediction = input('Enter a date to make a prediction on (YYYY-MM-DD):')
			#This line drives the try except
			predictionArr = X.loc[str(prediction)].tolist()
		except:
			print('Invalid date. Please enter another date')
			continue

		predictionArr2 = []
		predictionArr2.append(predictionArr)

		print('Predicting on date ' + colored(prediction, 'green'))
		
		modelPredictions = model.predict(np.array(predictionArr2))

		for val in range(len(holdTimesStr)):
			print('Predicted gain in ' + holdTimesStr[val] + ' days: ' + str(modelPredictions[0][val]))

		answer = input('Would you like to find another date? [Y/n]')
		if(answer.lower() == 'y'):
			predicting = True
		else:
			predicting = False

	

model = model(inputs,outputs)
makePrediction(model, X)