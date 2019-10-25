import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def print_acc(trainAcc, testAcc, dp, c1, c2,xAxis,file_name):
    
    #Converting lists to numpy arrays
    trainAcc = np.asarray(trainAcc)
    testAcc  = np.asarray(testAcc)
    
    #Plotting graph
    plt.plot(dp, trainAcc, c1, dp, testAcc, c2)
    plt.legend(['Training Accuracy','Test Accuracy'])
    plt.xlabel(xAxis)
    plt.title(file_name)
    plt.ylabel('Accuracy')
    plt.savefig(file_name + '.jpg')
    plt.show()


def des_tree(x_train, x_test, y_train, y_test, depth, criterion_name ):
    #making tree
    tr = tree.DecisionTreeClassifier(criterion=criterion_name, max_depth=depth)
    
    #Training tree
    tr = tr.fit(x_train, y_train)
    y_predTrain = tr.predict(x_train)
    y_predTest  = tr.predict(x_test)
    
    
    return metrics.accuracy_score(y_test,  y_predTest), metrics.accuracy_score(y_train, y_predTrain)
    

def k_neighbour(x_train, x_test, y_train, y_test, k, algo):
     
    #Creating tree and fitting data to tree
    tr =  KNeighborsClassifier(n_neighbors=k, metric=algo)
    tr.fit(x_train,y_train.values.ravel())
    
    
    #Predicting 
    y_predTrain = tr.predict(x_train)
    y_predTest  = tr.predict(x_test)
    
    
    return (accuracy_score(y_test, y_predTest),accuracy_score(y_train, y_predTrain))


def split_data(file):

    colNames = ['Sample Code #', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 
                'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class - 2=Benign 4=Malignant']
    file = pd.read_csv(file, header=None)
    file.columns = colNames

    train, test = train_test_split(file, test_size=0.2, random_state=2142)

    features = ['Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 
                'Single Epithelial Cell Size', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses']

    train.columns = colNames
    y_train = train['Class - 2=Benign 4=Malignant']
    x_train = train[features]

    test.columns = colNames
    y_test = test['Class - 2=Benign 4=Malignant']
    x_test = test[features]
    
    return x_train, x_test, y_train, y_test


def main():
    
    x_train, x_test, y_train, y_test = split_data('breast-cancer-wisconsin2.data')
    
    #
    #Trying decision tree with various depths from 2 to 25 with criterion entropy
    #
    dp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    trainAcc = []
    testAcc  = []
    print('Testing decision tree with Entropy as impurity measure')
    for i in dp:
       temp, temp2 = des_tree(x_train, x_test, y_train, y_test, i, 'entropy')
       trainAcc.append(temp)
       testAcc.append(temp2)
       
       
    #Printing graph
    print_acc(trainAcc, testAcc, dp, 'ko-','cv--', 'Depth', 'des_tree_entropy')


    #
    #Trying decision tree with various depths from 2 to 25 with criterion gini
    #
    trainAcc = []
    testAcc  = []
    print()
    print('Testing desicion tree with Gini as impurity measure')
    for i in dp:
       temp, temp2 = des_tree(x_train, x_test, y_train, y_test, i, 'gini')
       trainAcc.append(temp)
       testAcc.append(temp2)
    
    #Printing graph
    print_acc(trainAcc, testAcc, dp, 'ko-','cv--', 'Depth', 'des_tree_gini')



    #
    #Testing KNN with various neighbours from 2 to 25 with different distance metrics
    #
    print()
    colors =['ro-', 'go-', 'ko-', 'cv--', 'yv--', 'bv--']
    dist = ['euclidean', 'manhattan', 'cosine']
    print('-----------Testing k-nearest neighbor classifier-----------')
    count = 0
    for j in dist:
        trainAcc = []
        testAcc  = []
        print('Using', j, 'distance metric.')
        for i in dp:
            temp, temp2 = k_neighbour(x_train, x_test, y_train, y_test, i, j)
            trainAcc.append(temp)
            testAcc.append(temp2)
        print()
        #Printing graph
        print_acc(trainAcc, testAcc, dp, colors[count],colors[5 - count], 'Number of neighbors', 'KNN_' + j)
        count += 1




if __name__ == '__main__':
    main()





