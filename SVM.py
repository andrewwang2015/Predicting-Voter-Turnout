import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, svm
import math
import random
import scipy
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split


def loadData(fileName):
    '''
    Takes filename string and returns raw data as numpy array
    '''
    with open(fileName, 'r') as dest_f:
        data_iter = csv.reader(dest_f, delimiter = ',', quotechar = '"')
        data = [data for data in data_iter]
    rawData = np.asarray(data[1:], dtype = float)    # removes the first row of headers
    columnsToDelete = returnColumnsToDelete(rawData)
    processedData = deleteColumns(columnsToDelete, rawData)
    return processedData

def returnTrainingAndValidation(arr, percentageValidation):
    '''
    Splits all input data into a trainnig arr and validation arr according to
    percentage specified
    '''
    threshold = int(percentageValidation * len(arr))
    validation = arr[:threshold]
    training = arr[threshold:]
    return training, validation
    
def deleteColumns(columns, arr):
    '''
    Deletes particular columns in arrays 
    '''
    arr1 = scipy.delete(arr, columns, 1)
    return arr1

def returnColumnsToDelete(arr):
    columnsToDelete = [0]   #We want to delete column 0, or ID column to begin with
    for i in range(arr.shape[1]):
        currentCol = arr[:,i]
        if isDrop(currentCol):
            columnsToDelete.append(i)
    return columnsToDelete
        

def isDrop(arr):
    '''
    Determines which columns/ features to drop. If all column values are the
    same or most column values (>= 90% are <=0), then we drop
    '''
    isSame = 0
    numZeroOrLess = 0
    for i in arr:
        if i <= 0:
            numZeroOrLess += 1
        if i == arr[0]:
            isSame += 1
    if (isSame >= 0.9 * len(arr) or numZeroOrLess >= 0.9 * len(arr)):
        return True
    return False
            
def getInputsAndOutputs(arr):
    '''
    This function takes the raw data and returns two arrays for inputs(x) 
    and outputs(y) without x_0 being 1
    '''    
    inputs = []
    outputs = []

    for i in arr:
        inputs.append(i[:-1])   # the x vector
        outputs.append(i[(len(i) - 1)])      # last element, y
    return np.asarray(inputs), np.asarray(outputs)

def runSVM(inputs, outputs):
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array. 
    
    '''
    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(inputs, outputs)
    return clf

def main():
    allData = loadData("train_2008.csv")
    X, y = getInputsAndOutputs(allData)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    clf = runSVM(X_train, y_train)
    print(clf.score(X_test, y_test))
    

        

main()
