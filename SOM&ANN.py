# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom

# Get dataset, remove last class (wehter they have been accepted for a card or not)
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values

# Scale the data
sc = MinMaxScaler(feature_range = (0,1))
X = sc.fit_transform(X)

# Create SOM - adjust the final neurons here
som = MiniSom(10, 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Get the data ready for supervised learning
distancemap = som.distance_map().T
indexes = []
fraudulentrow = []

# In the distance map, look for every neuron that is above some threshould
for b in range (0, len(distancemap)):
    for h in range(0, len(distancemap[0])):
        if distancemap[b,h] > 0.93:
            indexes.append((h, b))

# Identify all customers that had that neuron as ther winning neuron
for z in range (0,len(X)):
    j = som.winner(X[z,:])
    for h in range(0, len(indexes)):
        if indexes[h] == j:
            fraudulentrow.append(z)

# Make a list of answers (to be used in supervised learning)
y = np.zeros(shape=(len(X),1))
for p in range (0, len(fraudulentrow)):
    y[fraudulentrow[p],:] = 1
    
# Make new data without the customer IDs, but including the last class
newX = dataset.iloc[:, 1:].values
newX = sc.fit_transform(newX)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newX, y, test_size = 0.2, random_state = 0)

# Import libraries
import keras 
from keras.models import Sequential
from keras.layers import Dense

# Set up ANN
classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu', input_dim = 15))
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation = 'relu'))
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
classifier.fit(X_train, y_train, batch_size = 2, epochs = 10)

# Predict, and set all to 1 above certain threshold
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# MAke a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)