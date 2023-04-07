import pandas as pd
import quandl,math, datetime, pickle
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

df = quandl.get('WIKI/GOOGL')
#reducing the number of columns
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#High-Low Percentage
df['HL_PCT'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100.0
#Percentage Change
df['PCT_change'] = (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100.0

#features
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
#marking invalid data as the outlier instead of getting rid of data
df.fillna(-99999, inplace=True)

#predicting 1% of the dataframe (using data that came 1% days ago to predict today)
forecast_out = int(math.ceil(0.01*len(df)))

#labels
df['labels'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['labels'],1)) #features = Everything-label

#scaling X before feeding it to classifier - Normalized with all the other datapoints
X = preprocessing.scale(X) # Add to processing time
X = X[:-forecast_out]
X_lately = X[-forecast_out:]

df.dropna(inplace=True)
y = np.array(df['labels']) #labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Linear Regression classifier
clf_lr = LinearRegression()
clf_lr.fit(X_train,y_train) #training

#pickling aka serializing a trained classifier
'''
with open('linear_regression.pickle','wb') as f:
    pickle.dump(clf_lr,f)

pickle_in = open('linear_regression.pickle','rb')
clf_lr = pickle.load(pickle_in)
'''


accuracy = clf_lr.score(X_test, y_test) #testing - Squared Error with LR
print(accuracy)

#SVM classifier
clf_svm = svm.SVR(kernel='poly')
clf_svm.fit(X_train,y_train) #training
accuracy = clf_svm.score(X_test, y_test) #testing - Squared Error with LR 
print(accuracy)

#predicting (35 = forecast_out) days values
forecast_set = clf_lr.predict(X_lately)

df['Forecast'] = np.nan #defining not a number data

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400 #number of seconds in the day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.head())
print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()