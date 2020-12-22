# Program to solve the Titanic - Machine Learning from Disaster Kaggle challenge
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

# Create a function to get data from a CSV file
# Returns a Pandas DataFrame
def getData(f):
    # Remove the name, ticket, and cabin information from the dataframe
    df = pd.read_csv(f).drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    return df

# Function that splits the data into the target and inputs
def splitData(arr):
    # X is the input
    X = arr[:, 1:]
    # Y is the target
    Y = arr[:, 0]

    print(X[0])
    print(Y[0])

    return (X, Y)

# Categories: [ID, Survival, Class, Sex, Age, Siblings/Spouses, Parents/Children, Ticket, Fare, Cabin, Port]
titanicTrain = getData('titanic/train.csv')
titanicTrain = pd.get_dummies(titanicTrain)
# print(titanicTrain.head())

# Convert the DF into a numpy array for use
titanicTrain = titanicTrain.to_numpy()
# print(titanicTrain[:10])

# Split the data into the target and the input
X_train, Y_train = splitData(titanicTrain)
print(X_train.shape)
print(Y_train.shape)

# A function that creates the model we'll use
def makeModel():
    model = Sequential()

    # Four layers, including input and output
    model.add(Input(shape=(10,)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    # model.summary()
    return model

model = makeModel()


# Fitting the model on the training data
history = model.fit(X_train, Y_train, batch_size=32, epochs=6, validation_split=.2)
