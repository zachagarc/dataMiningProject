import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import zscore
from sklearn.model_selection import cross_val_score


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
    plt.savefig(file_name + '.png')
    plt.show()


def des_tree(x_train, x_test, y_train, y_test, depth, criterion_name ):
    #making tree
    tr = tree.DecisionTreeClassifier(criterion=criterion_name, max_depth=depth)
    
    #Training tree
    tr = tr.fit(x_train, y_train)
    y_predTrain = tr.predict(x_train)
    y_predTest  = tr.predict(x_test)
    
    # Scoring classifier
    cvscore = cross_val_score(tr, x_train, y_train, cv=10,scoring='accuracy')
    
    return cvscore.mean(),cvscore, metrics.accuracy_score(y_test,  y_predTest), metrics.accuracy_score(y_train, y_predTrain)
    

def k_neighbour(x_train, x_test, y_train, y_test, k, algo):
     
    #Creating tree and fitting data to tree
    tr =  KNeighborsClassifier(n_neighbors=k, metric=algo)
    tr.fit(x_train,y_train.values.ravel())
    
    
    #Predicting 
    y_predTrain = tr.predict(x_train)
    y_predTest  = tr.predict(x_test)
    
    cvscore = cross_val_score(tr, x_train, y_train, cv=10,scoring='accuracy')
    
    return (cvscore.mean(),cvscore, accuracy_score(y_test, y_predTest), accuracy_score(y_train, y_predTrain))


def split_data(file):
    
    # Opening csv
    colNames = ['Sample Code #', 'Clump Thickness', 'Uniformity of Cell Size',
                'Uniformity of Cell Shape', 'Marginal Adhesion', 
                'Single Epithelial Cell Size', 'BareNuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class - 2=Benign 4=Malignant']
    file = pd.read_csv(file, header=None)
    file.columns = colNames
    
    # Getting info on dataset
    file = file.replace('?',np.NaN)
    print(file.info(), '\n')
    print('Finding number of null values\n', file.isnull().sum())
    
    # Q1A
    # Removing rows with null values
    file = file.dropna(axis=0, how='any')
    print('\nAll null values removed')
    print('Shape of file', file.shape, '\n')
    
    #Changeing type of BareNucli from Obj to int
    file['BareNuclei'] = file['BareNuclei'].astype(int)
    print(file.info(), '\n')
    
    # Removing outliers
    print(file.shape, 'Before')
    z=np.abs(zscore(file))
    outlier_row3,outlier_col3=np.where(z>=3) 
    outlier_row_3,outlier_col_3=np.where(z<=-3)
    dataset=file.drop(file.index[outlier_row3])
    dataset=file.drop(file.index[outlier_row_3])
    file=dataset.reset_index(drop=True)
    print(file.shape, 'After')
    print(file.isnull().sum())
    
    # Splitting data
    train, test = train_test_split(file,test_size=0.2, random_state=2142)

    # Excluding Sample Code and Class from features list
    features = colNames[1:10]
    
    train.columns = colNames
    y_train = train['Class - 2=Benign 4=Malignant']
    x_train = train[features]

    test.columns = colNames
    y_test = test['Class - 2=Benign 4=Malignant']
    x_test = test[features]
    
    return x_train, x_test, y_train, y_test

def main():
    
    x_train, x_test, y_train, y_test = split_data('breast-cancer-wisconsin.data')
    scores = []
    names = ['DTEntropy', 'DTGini', 'KNNEuclidean', 'KNNManhattan', 'KNNCosine']
    
    #
    #Trying decision tree with various depths from 2 to 25 with criterion entropy
    #
    dp = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
    trainAcc = []
    testAcc  = []
    mx_score = [0,0,[]]
    print('Testing decision tree with Entropy as impurity measure')
    for i in dp:
       cvscore,cvscores_arr, temp, temp2 = des_tree(x_train, x_test, y_train, y_test, i, 'entropy')
       trainAcc.append(temp2)
       testAcc.append(temp)
       if cvscore > mx_score[0]:
           mx_score[0] = cvscore
           mx_score[1] = i
           mx_score[2] = cvscores_arr
       
    #Printing graph
    print_acc(trainAcc, testAcc, dp, 'ko-','cv--', 'Depth', 'des_tree_entropy')
    print('Highest cross validation score is %0.4f'% mx_score[0], 'at depth', mx_score[1])
    scores.append(mx_score[2])
    
    #
    #Trying decision tree with various depths from 2 to 25 with criterion gini
    #
    trainAcc = []
    testAcc  = []
    mx_score = [0,0,[]]
    print()
    print('Testing desicion tree with Gini as impurity measure')
    for i in dp:
       cvscore, cvscores_arr, temp, temp2 = des_tree(x_train, x_test, y_train, y_test, i, 'gini')
       trainAcc.append(temp2)
       testAcc.append(temp)
       if cvscore > mx_score[0]:
           mx_score[0] = cvscore
           mx_score[1] = i
           mx_score[2] = cvscores_arr
    #Printing graph
    print_acc(trainAcc, testAcc, dp, 'ko-','cv--', 'Depth', 'des_tree_gini')
    print('Highest cross validation score is %0.4f'% mx_score[0], 'at depth', mx_score[1])
    scores.append(mx_score[2])


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
        mx_score = [0,0,[]]
        print('Using', j, 'distance metric.')
        for i in dp:
            cvscore, cvscores_arr, test, train = k_neighbour(x_train, x_test, y_train, y_test, i, j)
            trainAcc.append(train)
            testAcc.append(test)
            if cvscore > mx_score[0]:
                mx_score[0] = cvscore
                mx_score[1] = i
                mx_score[2] = cvscores_arr
            
        print()
        #Printing graph
        print_acc(trainAcc, testAcc, dp, colors[count],colors[5 - count], 'Number of neighbors', 'KNN_' + j)
        print('Highest cross validation score is %0.4f'% mx_score[0], 'at depth', mx_score[1])
        scores.append(mx_score[2])
        count += 1
    
    # boxplot algorithm comparison
    fig = plt.figure(figsize=(8,5))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(scores)
    ax.set_xticklabels(names)
    plt.savefig('AlgorithmComparison.png')
    plt.show()

if __name__ == '__main__':
    main()





