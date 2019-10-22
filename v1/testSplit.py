"""
Ivan Nieto
CS 488
"""
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


import matplotlib.pyplot as plt

def split_data(file_name):

    #reading files without addng column names 
    column_names = ['id','Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape','Marginal Adhesion', 
                    'Single Epithelial Cell Size', 'Bare Nuclei', 
                    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    
    data = pd.read_csv('breast-cancer-wisconsin.data', delimiter = ',',
                       encoding = 'utf-8-sig',
                       header = None, names = column_names)

    print(data.head())

    #Splitting data 80 20
    train, test = train_test_split(data, test_size=0.2,random_state=2142)
    
    #Creating files 
    create_csv(train, test, 'training.csv', 'testing.csv')
    
    
    features = ['Clump Thickness', 'Uniformity of Cell Size',
                    'Uniformity of Cell Shape','Marginal Adhesion', 
                    'Single Epithelial Cell Size', 'Bare Nuclei', 
                    'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
    x = data[features]
    y = data.Class
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
    
    #Creating files
    create_csv(x_train, x_test, 'x_train.csv', 'x_test.csv')
    create_csv(y_train, y_test, 'y_train.csv', 'y_test.csv')
    
    
def create_csv(li1, li2, nameli1, nameli2):
    np.savetxt(nameli1, li1, delimiter=',', fmt='%s')
    np.savetxt(nameli2, li2, delimiter=',', fmt='%s')
    
    
def des_tree(x_train, x_test, y_train, y_test, depth, criterion_name):
   
    
    x_train = pd.read_csv(x_train, header=None)
    x_train = [int(i) for i in x_train] 
    y_train = pd.read_csv(y_train, header=None)
    y_train = [int(i) for i in y_train] 
    x_test =  pd.read_csv(x_test,  header=None)
    x_test = [int(i) for i in x_test] 
    y_test =  pd.read_csv(y_test,  header=None)
    y_test = [int(i) for i in y_test] 
    
    print(y_test)
    print(y_train)
    print(x_train)
    print(x_test)
    
    #making tree
    tr = tree.DecisionTreeClassifier(criterion=criterion_name, max_depth=depth)
    
    
    #Training tree
    tr = tr.fit(x_train, y_train)
    y_predTrain = tr.predict(x_train)
    y_predTest  = tr.predict(x_test)
    
    
    return (metrics.accuracy_score(y_test,  y_predTest), 
            metrics.accuracy_score(y_train, y_predTrain))
    
def k_neighbour(x_train, x_test, y_train, y_test, k, algo):
    
    #Getting data
    x_train = pd.read_csv(x_train, header=None)
    y_train = pd.read_csv(y_train, header=None)
    x_test =  pd.read_csv(x_test,  header=None)
    y_test =  pd.read_csv(y_test,  header=None)
    
    tr =  KNeighborsClassifier(n_neighbors=k, metric=algo)
    tr.fit(x_train,y_train.values.ravel())
    
    
    y_predTrain = tr.predict(x_train)
    y_predTest  = tr.predict(x_test)
    
    
    return (accuracy_score(y_test, y_predTest),accuracy_score(y_train, y_predTrain))
    
def print_acc(trainAcc, testAcc, dp, c1, c2):
    
    #Converting lists to numpy arrays
    trainAcc = np.asarray(trainAcc)
    testAcc  = np.asarray(testAcc)
    
    #Plotting graph
    plt.plot(dp, trainAcc, c1, dp, testAcc, c2)
    plt.legend(['Training Accuracy','Test Accuracy'])
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()



def main():
    csv_f      = 'breast-cancer-wisconsin.csv'
    x_training = 'x_train.csv'
    x_testing  = 'x_test.csv'
    y_training = 'y_train.csv'
    y_testing  = 'y_test.csv'
    dp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    
     
    #1----------------------------------------------------
    split_data(csv_f)
    
    
    des_tree(x_training, x_testing, y_training, y_testing, 5, 'entropy')
    
    '''
    #2a----------------------------------------------------
    trainAcc = []
    testAcc  = []
    print('Testing decision tree with Entropy as impurity measure')
    for i in dp:
       temp, temp2 = des_tree(x_training, x_testing, y_training, y_testing, i, 'entropy')
       print('Accuracy with depth =', i, 'is', temp, 'for training data and', temp2, 'for test data.')
       trainAcc.append(temp)
       testAcc.append(temp2)
       
       
    #Printing graph
    print_acc(trainAcc, testAcc, dp, 'ko-','cv--' )
        
    
        
    #2b----------------------------------------------------
    trainAcc = []
    testAcc  = []
    print()
    print('Testing desicion tree with Gini as impurity measure')
    for i in dp:
       temp, temp2 = des_tree(x_training, x_testing, y_training, y_testing, i, 'gini')
       print('Accuracy with depth =', i, 'is', temp, 'for training data and', temp2, 'for test data.')
       trainAcc.append(temp)
       testAcc.append(temp2)
    
    #Printing graph
    print_acc(trainAcc, testAcc, dp, 'ko-','cv--' )
    
    
    
    #3----------------------------------------------------
    print()
    colors =['ro-', 'go-', 'ko-', 'cv--', 'yv--', 'bv--']
    dist = ['euclidean', 'manhattan', 'cosine']
    print('Testing k-nearest neighbor classifier')
    count = 0
    for j in dist:
        trainAcc = []
        testAcc  = []
        print('Using', j, 'distance metric.')
        for i in dp:
            temp, temp2 = k_neighbour(x_training, x_testing, y_training, y_testing, i, j)
            trainAcc.append(temp)
            testAcc.append(temp2)
            print('Accuracy with k =', i, 'is', temp, 'for training data and', temp2, 'for test data.')
        print()
        #Printing graph
        print_acc(trainAcc, testAcc, dp, colors[count],colors[5 - count] )
        count += 1
    '''
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    