import csv
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, svm
import math
import random
import scipy
from sklearn.svm import NuSVC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
deletionColumns = []


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
    global deletionColumns
    columnsToDelete = [0]   #We want to delete column 0, or ID column to begin with
    for i in range(arr.shape[1]):
        currentCol = arr[:,i]
        if isDrop(currentCol):
            columnsToDelete.append(i)
    if deletionColumns == []:
        deletionColumns = columnsToDelete
    return deletionColumns

        
def saveToCSV(fileName, arr):
    np.savetxt(fileName, arr, delimiter=",")
    
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

def runBagging(inputs, outputs, estimators, maxSamples, maxFeatures):
    
    '''
    This function takes in inputs and outputs of a dataset and returns
    the decision boundary of classifier, and predicted values as array. 
    
    '''
    bagging = BaggingClassifier(n_estimators = estimators, 
                                max_samples = maxSamples, 
                                max_features = maxFeatures)    
    bagging.fit(inputs, outputs)
    return bagging

def main():
    allData = loadData("train_2008.csv")
    X, y = getInputsAndOutputs(allData)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)    
    data2008 = loadData("test_2008.csv")
    
    #data2012 = loadData("test_2012.csv")
    #testing2012 = data2012
    
    #clf = runBagging(X, y, 60, 0.55, 1.0)
    
    #print(clf.score(X_test, y_test))
    #testPredictions = clf.predict(testing2012)
    #testPredictions = np.asarray(testPredictions, dtype = int)
    #saveToCSV("bagging2012_V4.csv", testPredictions)
    
    #pureTesting = loadData("test_2008.csv")
    #testingX = pureTesting
    

    #print(testing2012.shape)
    

    
    ### TESTING NUMBER OF ESTIMATORS ## 
    #numEstimatorsData = []
    #for numEstimators in range(40, 100, 5):
        #clf = runBagging(X_train, y_train, numEstimators, 1.0, 1.0)
        #numEstimatorsData.append(np.asarray([numEstimators,clf.score(X_test, y_test)]))
    #numEstimatorsData = np.asarray(numEstimatorsData)
    #for i in numEstimatorsData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Number of Base Estimators', fontsize = 22)    
    #plt.plot(numEstimatorsData[:,0], numEstimatorsData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Base Estimators', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       
    
    
    ## TESTING NUMBER OF SAMPLES
    #maxSamplesData = []
    #for maxSamples in np.linspace(0.1, 1.0, num=10):
        #clf = runBagging(X_train, y_train, 10, maxSamples, 1.0)
        #maxSamplesData.append(np.asarray([maxSamples,clf.score(X_test, y_test)]))
    #maxSamplesData = np.asarray(maxSamplesData)
    #for i in maxSamplesData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Number of Samples', fontsize = 22)    
    #plt.plot(maxSamplesData[:,0], maxSamplesData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Samples', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       
    
    
    ## TESTING NUMBER OF MAX FEATURES
    #maxFeaturesData = []
    #for maxFeatures in np.linspace(0.1, 1.0, num=10):
        #clf = runBagging(X_train, y_train, 10, 1.0 , maxFeatures)
        #maxFeaturesData.append(np.asarray([maxFeatures,clf.score(X_test, y_test)]))
    #maxFeaturesData = np.asarray(maxFeaturesData)
    #for i in maxFeaturesData:
        #print (i)
    
    #fig = plt.figure()
    #plt.title('Validation Accuracy vs. Number of Features', fontsize = 22)    
    #plt.plot(maxFeaturesData[:,0], maxFeaturesData[:,1], marker = '.', linewidth = 2)
    #plt.xlabel('Number of Features', fontsize = 18)
    #plt.ylabel('Accuracy/ Score', fontsize = 18)
    #plt.margins(y=0.02)       

    #clf = runBagging(X, y, 37, 0.7, 0.9)

    #testPredictions = clf.predict(testing2012)
    #testPredictions = np.asarray(testPredictions, dtype = int)
    #saveToCSV("bagging2012.csv", testPredictions)

    #allPossible = []
    #maximum = -999
    #for i in range(64,67):
        #for j in np.linspace(0.1, 1.0, num=5):
            #for k in np.linspace(0.1, 1.0, num=5):
                #clf = runBagging(X_train, y_train, i, j , k)
                #accuracy = clf.score(X_test, y_test)
                #if accuracy > maximum:
                    #maximum = accuracy
                    #print (i, j, k, maximum)
                #allPossible.append(np.asarray([i, j, k,accuracy]))
    #allPossible = np.asarray(allPossible)
    #for i in allPossible:
        #print(i)
    #print(maximum)
    
    #clf = runBagging(X, y, 60, 0.55, 1.0)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print (scores.mean())        
    
    clf = runBagging(X, y, 66, 0.775, 0.775)
    scores = cross_val_score(clf, X, y, cv=5)
    print (scores.mean())         
    testPredictions = clf.predict(data2008)
    testPredictions = np.asarray(testPredictions)
    saveToCSV("randomForestsFinalBagging.csv", testPredictions)    

main()
